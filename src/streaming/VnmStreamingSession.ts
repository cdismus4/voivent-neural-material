/**
 * VNM1 Streaming Session (Ticket D1-B)
 *
 * Manages the phased loading and progressive display lifecycle for an individual mesh/material:
 * - Progression sequence: Fallback -> Verified Tail -> Level 0 Neural Material.
 * - Flicker-free transitions: Current visual state is preserved until the next tier is 100% ready.
 * - Generation counter: Invalidates outdated or cancelled async results upon asset switch or disposal.
 * - Dual residency: Retains previous asset during cross-fade/switch until new tier is active.
 * - Decouples material attachment from GPU resource allocation.
 */

import type { NeuralMaterialManifest } from '../format/types';
import type { VnmChunkLoader } from './VnmChunkLoader';
import type { VnmResidencyManager } from './VnmResidencyManager';
import type {
  StreamingTier,
  ResidencyState,
  StreamingSessionCallbacks,
  ResidentNeuralAsset,
} from './types';

export class VnmStreamingSession {
  private readonly loader: VnmChunkLoader;
  private readonly residency: VnmResidencyManager;
  private readonly callbacks: StreamingSessionCallbacks;

  private currentTier: StreamingTier = 'fallback';
  private currentState: ResidencyState = 'unloaded';
  private currentAssetId: string | null = null;

  /** Monotonically increasing generation number to reject stale async responses */
  private generation = 0;

  /** Active abort controller for the current session loading pipeline */
  private abortController: AbortController | null = null;

  private isDisposed = false;

  constructor(
    loader: VnmChunkLoader,
    residency: VnmResidencyManager,
    callbacks: StreamingSessionCallbacks = {},
  ) {
    this.loader = loader;
    this.residency = residency;
    this.callbacks = callbacks;
  }

  public getTier(): StreamingTier {
    return this.currentTier;
  }

  public getState(): ResidencyState {
    return this.currentState;
  }

  public getCurrentAsset(): ResidentNeuralAsset | undefined {
    return this.currentAssetId ? this.residency.getAsset(this.currentAssetId) : undefined;
  }

  public getGeneration(): number {
    return this.generation;
  }

  /**
   * Begins phased progressive loading of a new material manifest.
   * Maintains display of the current material until the new material reaches a viewable tier.
   */
  public async loadMaterial(manifest: NeuralMaterialManifest): Promise<void> {
    if (this.isDisposed) {
      throw new Error('Cannot load material on a disposed streaming session');
    }

    // 1. Advance generation counter and cancel previous pending requests
    this.generation++;
    const activeGen = this.generation;

    if (this.abortController) {
      this.abortController.abort();
    }
    this.abortController = new AbortController();
    const signal = this.abortController.signal;

    const previousAssetId = this.currentAssetId;
    const targetAssetId = manifest.assetId;

    this.currentState = 'requested';
    this.callbacks.onTierChanged?.(this.currentTier, 'requested');

    // 2. Check if target asset is already fully resident in residency manager
    const existing = this.residency.getAsset(targetAssetId);
    if (existing && existing.tier === 'level0' && existing.weights && existing.level0) {
      this.activateNewAsset(targetAssetId, previousAssetId, 'level0', activeGen);
      return;
    }

    try {
      // 3. Phase 1: Rapid Tail Preview Loading (~2KB)
      try {
        const tailPayload = await this.loader.loadTail(manifest, signal);
        if (this.generation === activeGen && !this.isDisposed) {
          this.residency.registerAssetTier(
            manifest,
            'tail',
            { tail: tailPayload },
            activeGen,
          );
          // If we are still on fallback, promote to tail preview immediately
          if (this.currentTier === 'fallback' || previousAssetId !== targetAssetId) {
            this.activateNewAsset(targetAssetId, previousAssetId, 'tail', activeGen);
          }
        }
      } catch (err) {
        // Tail load failure is non-fatal; continue to level0 attempt
        if (!signal.aborted) {
          this.callbacks.onError?.(err instanceof Error ? err : new Error(String(err)), 'tail');
        }
      }

      if (this.generation !== activeGen || this.isDisposed) return;

      // 4. Phase 2: Full Level 0 Loading (Weights + Multi-res Latents)
      const [weightsPayload, level0Payload] = await Promise.all([
        this.loader.loadWeights(manifest, signal),
        this.loader.loadLatents(manifest, 0, signal),
      ]);

      if (this.generation !== activeGen || this.isDisposed) return;

      // Register complete Level 0 asset in residency manager
      this.residency.registerAssetTier(
        manifest,
        'level0',
        { weights: weightsPayload, level0: level0Payload },
        activeGen,
      );

      // 5. Activate Level 0 and release dual-residency on previous asset
      this.activateNewAsset(targetAssetId, previousAssetId, 'level0', activeGen);
    } catch (err) {
      if (this.generation === activeGen && !this.isDisposed) {
        if (!signal.aborted) {
          this.currentState = 'failed';
          this.callbacks.onError?.(err instanceof Error ? err : new Error(String(err)), 'level0');
          this.callbacks.onTierChanged?.(this.currentTier, 'failed');
        }
      }
    }
  }

  /**
   * Safely activates the new asset tier, updating reference counts and release previous asset.
   */
  private activateNewAsset(
    newAssetId: string,
    previousAssetId: string | null,
    newTier: StreamingTier,
    targetGen: number,
  ): void {
    if (this.generation !== targetGen || this.isDisposed) return;

    // Retain new asset
    this.residency.retain(newAssetId);

    // If switching from a different asset, release the previous asset
    if (previousAssetId && previousAssetId !== newAssetId) {
      this.residency.release(previousAssetId);
    }

    this.currentAssetId = newAssetId;
    this.currentTier = newTier;
    this.currentState = 'active';

    const asset = this.residency.getAsset(newAssetId);
    if (asset) {
      (asset as { state: ResidencyState }).state = 'active';
    }

    this.callbacks.onTierChanged?.(newTier, 'active');
  }

  /**
   * Disposes the session, releasing all asset references and cancelling inflight requests.
   */
  public dispose(): void {
    if (this.isDisposed) return;
    this.isDisposed = true;
    this.generation++;

    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }

    if (this.currentAssetId) {
      this.residency.release(this.currentAssetId);
      this.currentAssetId = null;
    }

    this.currentState = 'unloaded';
    this.currentTier = 'fallback';
  }
}
