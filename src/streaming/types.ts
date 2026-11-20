/**
 * Voivent Neural Material (VNM1) Streaming & Residency Types (Ticket D1)
 * Clean-room architecture for phased progressive material loading and memory budgeting.
 */

import type {
  NeuralMaterialManifest,
  ChunkRef,
  MlpWeights,
  LatentFeatureGrid,
} from '../format/types';
import type { ParsedTailChunk } from '../format/binary';

/**
 * Phased material residency state.
 * - unloaded: Not loaded or fully evicted.
 * - requested: Async loading in progress.
 * - resident: Loaded in memory and validated, ready to bind.
 * - active: Currently bound to material and rendered on mesh.
 * - failed: Loading failed; fallback is displayed.
 */
export type ResidencyState = 'unloaded' | 'requested' | 'resident' | 'active' | 'failed';

/**
 * Progression tiers for phased rendering.
 * 1. fallback: Instant procedural/placeholder material.
 * 2. tail: Fast box-filtered low-resolution unorm PBR texture (~2KB).
 * 3. level0: Full neural material (weights + multi-res latent grids, ~25KB).
 */
export type StreamingTier = 'fallback' | 'tail' | 'level0';

export interface ChunkLoaderOptions {
  fetchFn?: (url: string, signal?: AbortSignal) => Promise<Uint8Array>;
  cacheGet?: (key: string) => Promise<Uint8Array | null>;
  cachePut?: (key: string, data: Uint8Array, expectedSha256: string) => Promise<boolean | void>;
  cacheDelete?: (key: string) => Promise<boolean | void>;
  nowFn?: () => number;
}

export interface ChunkFetchRequest {
  readonly chunkRef: ChunkRef;
  readonly chunkKind: number;
  readonly label?: string;
}

export interface ValidatedWeightsPayload {
  readonly weights: MlpWeights;
  readonly rawBytes: Uint8Array;
}

export interface ValidatedLatentsPayload {
  readonly fine: LatentFeatureGrid;
  readonly coarse: LatentFeatureGrid;
  readonly rawBytes: Uint8Array;
}

export interface ValidatedTailPayload {
  readonly tail: ParsedTailChunk;
  readonly rawBytes: Uint8Array;
}

export interface MemoryBudgetConfig {
  /** Maximum CPU memory (bytes) for cached/resident chunks. Default: 64MB */
  maxCpuBytes?: number;
  /** Maximum GPU memory (bytes) for textures/storage buffers. Default: 128MB */
  maxGpuBytes?: number;
  /** Maximum concurrent inflight download bytes. Default: 16MB */
  maxInflightBytes?: number;
}

export interface MemoryBudgetStats {
  cpuBytesUsed: number;
  gpuBytesUsed: number;
  inflightBytes: number;
  residentAssetCount: number;
  activeAssetCount: number;
}

export interface StreamingSessionCallbacks {
  onTierChanged?: (tier: StreamingTier, state: ResidencyState) => void;
  onError?: (error: Error, tier: StreamingTier) => void;
  onProgress?: (bytesLoaded: number, totalBytes: number) => void;
}

export interface ResidentNeuralAsset {
  readonly assetId: string;
  readonly manifest: NeuralMaterialManifest;
  readonly weights?: ValidatedWeightsPayload;
  readonly level0?: ValidatedLatentsPayload;
  readonly tail?: ValidatedTailPayload;
  readonly tier: StreamingTier;
  readonly state: ResidencyState;
  readonly generation: number;
  readonly cpuByteSize: number;
  readonly gpuByteSize: number;
  lastAccessedAt: number;
  referenceCount: number;
}
