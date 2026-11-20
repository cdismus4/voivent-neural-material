/**
 * VNM1 Verified Chunk Loader (Ticket D1-A)
 *
 * Provides cryptographic and structural validation for network & cached VNM chunks:
 * - SHA-256 integrity check matching manifest.
 * - Exact byte length and chunk header kind validation.
 * - Profile and dimension consistency checks with manifest.
 * - Corrupted cache recovery: automatically purges corrupt cache entry and re-fetches from network once.
 * - Inflight request coalescing: multiple consumers share single download promise without cross-cancellation.
 * - Dependency injection for headless, GPU-free unit testing.
 */

import {
  parseVnmHeader,
  parseWeightsChunk,
  parseLatentsChunk,
  parseTailChunk,
  VnmChunkKind,
} from '../format/binary';
import type {
  NeuralMaterialManifest,
  ChunkRef,
  LatentLevel,
} from '../format/types';
import {
  getCachedChunk,
  putCachedChunk,
  deleteCachedChunk,
  verifyChunkSha256Status,
} from '../runtime/PersistentChunkCache';
import type {
  ChunkLoaderOptions,
  ValidatedWeightsPayload,
  ValidatedLatentsPayload,
  ValidatedTailPayload,
} from './types';

interface InflightRequest {
  readonly promise: Promise<Uint8Array>;
  readonly abortController: AbortController;
  readonly subscribers: Set<AbortSignal | undefined>;
}

export class VnmChunkLoader {
  private readonly fetchFn: (url: string, signal?: AbortSignal) => Promise<Uint8Array>;
  private readonly cacheGet: (key: string) => Promise<Uint8Array | null>;
  private readonly cachePut: (key: string, data: Uint8Array, expectedSha256: string) => Promise<boolean | void>;
  private readonly cacheDelete: (key: string) => Promise<boolean | void>;
  private readonly nowFn: () => number;

  /** Active inflight chunk requests keyed by URL */
  private readonly inflightRequests = new Map<string, InflightRequest>();

  constructor(options: ChunkLoaderOptions = {}) {
    this.fetchFn = options.fetchFn ?? VnmChunkLoader.defaultFetch;
    this.cacheGet = options.cacheGet ?? getCachedChunk;
    this.cachePut = options.cachePut ?? putCachedChunk;
    this.cacheDelete = options.cacheDelete ?? deleteCachedChunk;
    this.nowFn = options.nowFn ?? (() => Date.now());
  }

  public getNow(): number {
    return this.nowFn();
  }

  private static async defaultFetch(url: string, signal?: AbortSignal): Promise<Uint8Array> {
    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText} fetching chunk: ${url}`);
    }
    const buf = await response.arrayBuffer();
    return new Uint8Array(buf);
  }

  /**
   * Loads and validates the weights.bin chunk for a given manifest.
   */
  public async loadWeights(
    manifest: NeuralMaterialManifest,
    signal?: AbortSignal,
  ): Promise<ValidatedWeightsPayload> {
    const ref = manifest.weights;
    const raw = await this.loadAndValidateRawChunk(
      ref,
      VnmChunkKind.Weights,
      'weights',
      signal,
    );

    const weights = parseWeightsChunk(raw, manifest.profile);
    return { weights, rawBytes: raw };
  }

  /**
   * Loads and validates a latent level chunk (e.g. level-0.bin).
   */
  public async loadLatents(
    manifest: NeuralMaterialManifest,
    levelIndex = 0,
    signal?: AbortSignal,
  ): Promise<ValidatedLatentsPayload> {
    const level: LatentLevel | undefined = manifest.levels[levelIndex];
    if (!level) {
      throw new Error(`Manifest does not contain latent level index ${levelIndex}`);
    }

    const raw = await this.loadAndValidateRawChunk(
      level.chunk,
      VnmChunkKind.Latents,
      `level-${levelIndex}`,
      signal,
    );

    const parsed = parseLatentsChunk(raw, manifest.profile);

    // Dimension consistency verification with manifest
    if (
      parsed.fine.width !== level.fineSize[0] ||
      parsed.fine.height !== level.fineSize[1] ||
      parsed.coarse.width !== level.coarseSize[0] ||
      parsed.coarse.height !== level.coarseSize[1]
    ) {
      throw new Error(
        `Latent dimension mismatch with manifest: ` +
          `Expected Fine(${level.fineSize[0]}x${level.fineSize[1]}) Coarse(${level.coarseSize[0]}x${level.coarseSize[1]}), ` +
          `got Fine(${parsed.fine.width}x${parsed.fine.height}) Coarse(${parsed.coarse.width}x${parsed.coarse.height})`,
      );
    }

    return { fine: parsed.fine, coarse: parsed.coarse, rawBytes: raw };
  }

  /**
   * Loads and validates the tail.bin preview chunk.
   */
  public async loadTail(
    manifest: NeuralMaterialManifest,
    signal?: AbortSignal,
  ): Promise<ValidatedTailPayload> {
    const ref = manifest.tail;
    const raw = await this.loadAndValidateRawChunk(
      ref,
      VnmChunkKind.Tail,
      'tail',
      signal,
    );

    const tail = parseTailChunk(raw);
    if (tail.tailFirstMip !== manifest.tailFirstMip) {
      throw new Error(
        `Tail first mip mismatch with manifest: expected ${manifest.tailFirstMip}, got ${tail.tailFirstMip}`,
      );
    }
    return { tail, rawBytes: raw };
  }

  /**
   * Core loading routine: Deduplicates simultaneous requests per URL,
   * checks cache -> validates -> (fetches on network if missed/corrupted) -> caches.
   */
  private async loadAndValidateRawChunk(
    ref: ChunkRef,
    expectedKind: VnmChunkKind,
    label: string,
    signal?: AbortSignal,
  ): Promise<Uint8Array> {
    if (signal?.aborted) {
      throw new DOMException('Operation aborted', 'AbortError');
    }

    const url = ref.url;
    let inflight = this.inflightRequests.get(url);

    if (!inflight) {
      const abortController = new AbortController();
      const subscribers = new Set<AbortSignal | undefined>();

      const runLoader = async (): Promise<Uint8Array> => {
        // 1. Try reading from persistent cache first
        try {
          const cached = await this.cacheGet(url);
          if (cached) {
            const isValid = await this.validateChunkData(cached, ref, expectedKind);
            if (isValid) {
              return cached;
            }
            // Cache corrupted: purge entry and fall through to network fetch
            console.warn(`[VnmChunkLoader] Corrupted cache detected for ${url}. Evicting and refetching.`);
            await this.cacheDelete(url);
          }
        } catch {
          // Cache read failure is non-fatal; continue to network fetch
        }

        // Check if aborted before network fetch
        if (abortController.signal.aborted) {
          throw new DOMException('Operation aborted', 'AbortError');
        }

        // 2. Fetch from network
        const networkData = await this.fetchFn(url, abortController.signal);

        // 3. Validate network data strictly
        const isValid = await this.validateChunkData(networkData, ref, expectedKind);
        if (!isValid) {
          throw new Error(
            `[VnmChunkLoader] Network chunk validation failed for ${label} (${url}): ` +
              `expected SHA-256 ${ref.sha256}, expected length ${ref.byteLength}`,
          );
        }

        // 4. Store validated chunk in cache asynchronously
        try {
          await this.cachePut(url, networkData, ref.sha256);
        } catch {
          // Cache write failure is non-fatal
        }

        return networkData;
      };

      const promise = runLoader().finally(() => {
        this.inflightRequests.delete(url);
      });
      // Attach silent rejection handler to prevent unhandled rejection in background if all abort
      promise.catch(() => {});

      inflight = {
        promise,
        abortController,
        subscribers,
      };
      this.inflightRequests.set(url, inflight);
    }

    // Register this caller's signal
    const currentInflight = inflight;
    currentInflight.subscribers.add(signal);

    return new Promise<Uint8Array>((resolve, reject) => {
      let cleanup = () => {};

      if (signal) {
        const onAbort = () => {
          cleanup();
          currentInflight.subscribers.delete(signal);

          // Only abort underlying loader if NO active subscribers remain
          let anyActive = false;
          for (const s of currentInflight.subscribers) {
            if (!s || !s.aborted) {
              anyActive = true;
              break;
            }
          }

          if (!anyActive) {
            currentInflight.abortController.abort();
            this.inflightRequests.delete(url);
          }
          reject(new DOMException('Operation aborted', 'AbortError'));
        };

        if (signal.aborted) {
          onAbort();
          return;
        }

        signal.addEventListener('abort', onAbort, { once: true });
        cleanup = () => {
          signal.removeEventListener('abort', onAbort);
        };
      }

      currentInflight.promise.then(
        (data) => {
          cleanup();
          currentInflight.subscribers.delete(signal);
          resolve(data);
        },
        (err) => {
          cleanup();
          currentInflight.subscribers.delete(signal);
          reject(err);
        },
      );
    });
  }

  /**
   * Performs cryptographic (SHA-256), length, and VNM header validation on raw bytes.
   */
  private async validateChunkData(
    data: Uint8Array,
    ref: ChunkRef,
    expectedKind: VnmChunkKind,
  ): Promise<boolean> {
    // 1. Byte length check
    if (data.byteLength !== ref.byteLength) {
      return false;
    }

    // 2. Cryptographic SHA-256 check (SPEC §3 requires verification before use; reject 'mismatch' and 'unavailable')
    const status = await verifyChunkSha256Status(data, ref.sha256);
    if (status !== 'match') {
      return false;
    }

    // 3. VNM Header check
    try {
      const header = parseVnmHeader(data);
      if (header.chunkKind !== expectedKind) {
        return false;
      }
      if (header.payloadByteLength !== ref.byteLength - 16) {
        return false;
      }
    } catch {
      return false;
    }

    return true;
  }
}
