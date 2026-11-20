/**
 * VNM1 Residency Manager (Ticket D1-B)
 *
 * Central coordinator for neural material memory budgets and asset lifecycle:
 * - Tracks resident CPU and GPU memory usage.
 * - LRU eviction of unreferenced resident assets when budget is exceeded.
 * - Ensures active (in-use) assets are never evicted.
 * - Accounts for temporary dual-residency during material cross-fades and switches.
 * - Decouples mesh material unbinding from GPU buffer/texture disposal.
 */

import {
  getLatentChannelCount,
  MLP_PROFILES,
  type NeuralMaterialManifest,
} from '../format/types';
import type {
  MemoryBudgetConfig,
  MemoryBudgetStats,
  ResidentNeuralAsset,
  StreamingTier,
  ResidencyState,
  ValidatedWeightsPayload,
  ValidatedLatentsPayload,
  ValidatedTailPayload,
} from './types';

export class VnmResidencyManager {
  public readonly maxCpuBytes: number;
  public readonly maxGpuBytes: number;
  public readonly maxInflightBytes: number;

  /** Active and resident assets keyed by assetId */
  private readonly assets = new Map<string, ResidentNeuralAsset>();

  private inflightBytes = 0;

  constructor(config: MemoryBudgetConfig = {}) {
    this.maxCpuBytes = config.maxCpuBytes ?? 64 * 1024 * 1024; // 64 MB
    this.maxGpuBytes = config.maxGpuBytes ?? 128 * 1024 * 1024; // 128 MB
    this.maxInflightBytes = config.maxInflightBytes ?? 16 * 1024 * 1024; // 16 MB
  }

  public getStats(): MemoryBudgetStats {
    let cpuBytesUsed = 0;
    let gpuBytesUsed = 0;
    let residentCount = 0;
    let activeCount = 0;

    for (const asset of this.assets.values()) {
      cpuBytesUsed += asset.cpuByteSize;
      gpuBytesUsed += asset.gpuByteSize;
      if (asset.state === 'resident' || asset.state === 'active') {
        residentCount++;
      }
      if (asset.state === 'active') {
        activeCount++;
      }
    }

    return {
      cpuBytesUsed,
      gpuBytesUsed,
      inflightBytes: this.inflightBytes,
      residentAssetCount: residentCount,
      activeAssetCount: activeCount,
    };
  }

  public getAsset(assetId: string): ResidentNeuralAsset | undefined {
    const asset = this.assets.get(assetId);
    if (asset) {
      asset.lastAccessedAt = Date.now();
    }
    return asset;
  }

  /**
   * Registers newly acquired tier data for an asset, computing memory usage and triggering eviction if needed.
   */
  public registerAssetTier(
    manifest: NeuralMaterialManifest,
    tier: StreamingTier,
    payloads: {
      weights?: ValidatedWeightsPayload;
      level0?: ValidatedLatentsPayload;
      tail?: ValidatedTailPayload;
    },
    generation: number,
  ): ResidentNeuralAsset {
    const assetId = manifest.assetId;
    let existing = this.assets.get(assetId);

    const weights = payloads.weights ?? existing?.weights;
    const level0 = payloads.level0 ?? existing?.level0;
    const tail = payloads.tail ?? existing?.tail;

    // Compute CPU bytes: raw binary chunks + JS structures
    let cpuBytes = 0;
    let gpuBytes = 0;

    if (weights) {
      cpuBytes += weights.rawBytes?.byteLength ?? 0;
      const profileWeightsBytes = MLP_PROFILES[manifest.profile]
        ? MLP_PROFILES[manifest.profile].totalWeights * 4
        : 4448;
      // Storage buffer size on GPU: totalWeights f32 = totalWeights * 4 bytes
      gpuBytes += weights.rawBytes?.byteLength ?? profileWeightsBytes;
    }
    if (level0) {
      cpuBytes += level0.rawBytes?.byteLength ?? 0;
      // Latent textures on GPU: Fine + Coarse RGBA8 (1 texture/grid for 4ch, 2 textures/grid for 8ch)
      const latentCh =
        level0.fine?.channels ??
        level0.coarse?.channels ??
        (MLP_PROFILES[manifest.profile] ? getLatentChannelCount(manifest.profile) : 4);
      const fw = level0.fine?.width ?? 16;
      const fh = level0.fine?.height ?? 16;
      const cw = level0.coarse?.width ?? 8;
      const ch = level0.coarse?.height ?? 8;
      gpuBytes += (fw * fh + cw * ch) * latentCh;
    }
    if (tail) {
      cpuBytes += tail.rawBytes?.byteLength ?? 0;
      // Tail preview on GPU allocates 3 RGBA8 DataTextures (baseColor, normal, ORM) = 12 bytes/texel
      const tw = tail.tail?.tailWidth ?? 16;
      const th = tail.tail?.tailHeight ?? 16;
      gpuBytes += tw * th * 12;
    }

    const state: ResidencyState = tier === 'fallback' ? 'unloaded' : 'resident';

    const asset: ResidentNeuralAsset = {
      assetId,
      manifest,
      weights,
      level0,
      tail,
      tier,
      state,
      generation,
      cpuByteSize: cpuBytes,
      gpuByteSize: gpuBytes,
      lastAccessedAt: Date.now(),
      referenceCount: existing?.referenceCount ?? 0,
    };

    this.assets.set(assetId, asset);

    // Enforce memory budget with LRU eviction
    this.enforceMemoryBudget();

    return asset;
  }

  /**
   * Retains an asset reference when bound to a mesh/session.
   */
  public retain(assetId: string): void {
    const asset = this.assets.get(assetId);
    if (asset) {
      asset.referenceCount++;
      asset.lastAccessedAt = Date.now();
    }
  }

  /**
   * Releases an asset reference.
   * If reference count reaches 0, the asset remains resident in memory (for fast re-use)
   * until LRU budget eviction decides to purge it.
   */
  public release(assetId: string): void {
    const asset = this.assets.get(assetId);
    if (asset && asset.referenceCount > 0) {
      asset.referenceCount--;
      asset.lastAccessedAt = Date.now();
      if (asset.referenceCount === 0 && asset.state === 'active') {
        (asset as { state: ResidencyState }).state = 'resident';
      }
    }
  }

  /**
   * Tracks inflight download byte volume.
   */
  public recordInflightDelta(deltaBytes: number): void {
    this.inflightBytes = Math.max(0, this.inflightBytes + deltaBytes);
  }

  public canStartInflight(bytes: number): boolean {
    return this.inflightBytes + bytes <= this.maxInflightBytes;
  }

  /**
   * Enforces CPU and GPU memory budgets using strict LRU eviction.
   * NEVER evicts assets that are currently active (referenceCount > 0).
   */
  public enforceMemoryBudget(): void {
    const stats = this.getStats();
    const cpuOk = Number.isFinite(stats.cpuBytesUsed) && stats.cpuBytesUsed <= this.maxCpuBytes;
    const gpuOk = Number.isFinite(stats.gpuBytesUsed) && stats.gpuBytesUsed <= this.maxGpuBytes;
    if (cpuOk && gpuOk) {
      return;
    }

    // Sort unreferenced assets by lastAccessedAt ascending (oldest first)
    const candidates = Array.from(this.assets.values())
      .filter((a) => a.referenceCount === 0)
      .sort((a, b) => a.lastAccessedAt - b.lastAccessedAt);

    for (const candidate of candidates) {
      const currentStats = this.getStats();
      const cOk = Number.isFinite(currentStats.cpuBytesUsed) && currentStats.cpuBytesUsed <= this.maxCpuBytes;
      const gOk = Number.isFinite(currentStats.gpuBytesUsed) && currentStats.gpuBytesUsed <= this.maxGpuBytes;
      if (cOk && gOk) {
        break;
      }

      // Evict candidate from residency
      this.assets.delete(candidate.assetId);
    }
  }

  /**
   * Explicitly purges all unreferenced assets.
   */
  public purgeUnreferenced(): void {
    for (const [id, asset] of this.assets.entries()) {
      if (asset.referenceCount === 0) {
        this.assets.delete(id);
      }
    }
  }
}
