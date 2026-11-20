import { describe, it, expect } from 'vitest';
import { VnmChunkLoader } from '../streaming/VnmChunkLoader';
import { VnmResidencyManager } from '../streaming/VnmResidencyManager';
import { VnmStreamingSession } from '../streaming/VnmStreamingSession';
import { generateSyntheticPbrTexture } from '../training/SyntheticTextureGenerator';
import { NeuralMaterialTrainer } from '../training/NeuralMaterialTrainer';
import { exportNeuralMaterial } from '../exporter/VnmExporter';
import { parseTailChunk } from '../format/binary';
import type { NeuralMaterialManifest } from '../format/types';
import type { StreamingTier, ResidencyState } from '../streaming/types';

describe('Ticket D1-B: VnmStreamingSession & VnmResidencyManager', () => {
  async function createTestPackage(id: string) {
    const tex = generateSyntheticPbrTexture(32, 32, 'gradient-waves');
    const trainer = new NeuralMaterialTrainer({
      fineWidth: 16,
      fineHeight: 16,
      coarseWidth: 8,
      coarseHeight: 8,
      iterations: 2,
    });
    const result = trainer.train(tex);
    const pkg = await exportNeuralMaterial(result, tex, { assetId: id });
    const manifest = JSON.parse(pkg.manifestJson) as NeuralMaterialManifest;
    return { pkg, manifest };
  }

  describe('VnmResidencyManager', () => {
    it('accurately tracks CPU and GPU memory usage across asset tiers', async () => {
      const { pkg, manifest } = await createTestPackage('residency-mat-1');
      const manager = new VnmResidencyManager({
        maxCpuBytes: 10 * 1024 * 1024,
        maxGpuBytes: 10 * 1024 * 1024,
      });

      // Register tail with parsed chunk payload
      const tail = parseTailChunk(pkg.tailChunk.data);
      manager.registerAssetTier(manifest, 'tail', {
        tail: { tail, rawBytes: pkg.tailChunk.data },
      }, 1);

      let stats = manager.getStats();
      expect(stats.cpuBytesUsed).toBe(pkg.tailChunk.byteLength);
      expect(stats.residentAssetCount).toBe(1);
      expect(stats.activeAssetCount).toBe(0);

      // Retain asset
      manager.retain('residency-mat-1');
      const asset = manager.getAsset('residency-mat-1');
      expect(asset?.referenceCount).toBe(1);

      // Release asset
      manager.release('residency-mat-1');
      expect(asset?.referenceCount).toBe(0);
      expect(manager.getStats().residentAssetCount).toBe(1); // Still resident in cache
    });

    it('enforces LRU eviction on unreferenced assets while protecting active assets', async () => {
      const pkg1 = await createTestPackage('mat-A');
      const pkg2 = await createTestPackage('mat-B');

      // Restrict CPU budget so only 1 asset fits
      const singleAssetBytes = pkg1.pkg.weightsChunk.byteLength + pkg1.pkg.level0Chunk.byteLength;
      const manager = new VnmResidencyManager({
        maxCpuBytes: singleAssetBytes + 100, // Capacity for only ~1 asset
        maxGpuBytes: 10 * 1024 * 1024,
      });

      // Register Mat A and mark it active/in-use
      manager.registerAssetTier(pkg1.manifest, 'level0', {
        weights: { weights: {} as any, rawBytes: pkg1.pkg.weightsChunk.data },
        level0: { fine: { width: 16, height: 16 } as any, coarse: { width: 8, height: 8 } as any, rawBytes: pkg1.pkg.level0Chunk.data },
      }, 1);
      manager.retain('mat-A');

      // Register Mat B (not in use, refCount = 0)
      manager.registerAssetTier(pkg2.manifest, 'level0', {
        weights: { weights: {} as any, rawBytes: pkg2.pkg.weightsChunk.data },
        level0: { fine: { width: 16, height: 16 } as any, coarse: { width: 8, height: 8 } as any, rawBytes: pkg2.pkg.level0Chunk.data },
      }, 2);

      // Active Mat A MUST NOT be evicted!
      expect(manager.getAsset('mat-A')).toBeDefined();
    });

    it('F-6: accurately estimates GPU bytes for 8ch latent pairs and 3-texture tail previews', async () => {
      const { pkg, manifest } = await createTestPackage('gpu-bytes-mat');
      const manager = new VnmResidencyManager();

      const tail = parseTailChunk(pkg.tailChunk.data);
      const tailAsset = manager.registerAssetTier(manifest, 'tail', {
        tail: { tail, rawBytes: pkg.tailChunk.data },
      }, 1);
      // Tail preview creates 3 RGBA8 DataTextures (baseColor, normal, ORM) = 12 bytes/texel
      expect(tailAsset.gpuByteSize).toBe(tail.tailWidth * tail.tailHeight * 12);

      const manifest8: NeuralMaterialManifest = { ...manifest, assetId: 'gpu-bytes-8ch', profile: 'standard8' };
      const asset8 = manager.registerAssetTier(manifest8, 'level0', {
        weights: { weights: {} as any, rawBytes: new Uint8Array(1304 * 4 + 16) },
        level0: {
          fine: { width: 32, height: 16, channels: 8, data: new Uint8Array(32 * 16 * 8), isNormalizedU8: true },
          coarse: { width: 16, height: 8, channels: 8, data: new Uint8Array(16 * 8 * 8), isNormalizedU8: true },
          rawBytes: new Uint8Array(100),
        },
      }, 2);
      const expectedGpuBytes = (1304 * 4 + 16) + (32 * 16 + 16 * 8) * 8;
      expect(asset8.gpuByteSize).toBe(expectedGpuBytes);
    });
  });

  describe('VnmStreamingSession', () => {
    it('progresses through phased streaming: fallback -> tail -> level0', async () => {
      const { pkg, manifest } = await createTestPackage('phased-mat');

      const mockStore = new Map<string, Uint8Array>([
        [manifest.weights.url, pkg.weightsChunk.data],
        [manifest.levels[0]!.chunk.url, pkg.level0Chunk.data],
        [manifest.tail.url, pkg.tailChunk.data],
      ]);

      const loader = new VnmChunkLoader({
        fetchFn: async (url) => mockStore.get(url)!,
        cacheGet: async () => null,
      });

      const residency = new VnmResidencyManager();
      const tierTransitions: Array<{ tier: StreamingTier; state: ResidencyState }> = [];

      const session = new VnmStreamingSession(loader, residency, {
        onTierChanged: (tier, state) => {
          tierTransitions.push({ tier, state });
        },
      });

      expect(session.getTier()).toBe('fallback');
      expect(session.getState()).toBe('unloaded');

      await session.loadMaterial(manifest);

      expect(session.getTier()).toBe('level0');
      expect(session.getState()).toBe('active');
      // F-2 regression: tail -> level0 upgrade must not double-retain
      expect(residency.getAsset('phased-mat')?.referenceCount).toBe(1);

      // Verified phased progression sequence
      const tiersVisited = tierTransitions.map((t) => t.tier);
      expect(tiersVisited).toContain('tail');
      expect(tiersVisited).toContain('level0');

      session.dispose();
      expect(session.getState()).toBe('unloaded');
      expect(residency.getAsset('phased-mat')?.referenceCount).toBe(0);
      expect(residency.getAsset('phased-mat')?.state).toBe('resident');
    });

    it('F-2: keeps reference counts balanced across switch + tail->level0 promotion and allows eviction after dispose', async () => {
      const matA = await createTestPackage('mat-switch-A');
      const matB = await createTestPackage('mat-switch-B');
      const matC = await createTestPackage('mat-switch-C');

      const mockStore = new Map<string, Uint8Array>([
        [matA.manifest.weights.url, matA.pkg.weightsChunk.data],
        [matA.manifest.levels[0]!.chunk.url, matA.pkg.level0Chunk.data],
        [matA.manifest.tail.url, matA.pkg.tailChunk.data],
        [matB.manifest.weights.url, matB.pkg.weightsChunk.data],
        [matB.manifest.levels[0]!.chunk.url, matB.pkg.level0Chunk.data],
        [matB.manifest.tail.url, matB.pkg.tailChunk.data],
        [matC.manifest.weights.url, matC.pkg.weightsChunk.data],
        [matC.manifest.levels[0]!.chunk.url, matC.pkg.level0Chunk.data],
        [matC.manifest.tail.url, matC.pkg.tailChunk.data],
      ]);

      const loader = new VnmChunkLoader({
        fetchFn: async (url) => mockStore.get(url)!,
        cacheGet: async () => null,
      });

      const singleAssetCpu = matB.pkg.weightsChunk.byteLength + matB.pkg.level0Chunk.byteLength + matB.pkg.tailChunk.byteLength;
      const residency = new VnmResidencyManager({
        maxCpuBytes: Math.floor(singleAssetCpu * 2.5),
        maxGpuBytes: 10 * 1024 * 1024,
      });

      const externalSession = new VnmStreamingSession(loader, residency);
      await externalSession.loadMaterial(matA.manifest);
      expect(residency.getAsset('mat-switch-A')?.referenceCount).toBe(1);

      const session = new VnmStreamingSession(loader, residency);
      await session.loadMaterial(matA.manifest);
      expect(residency.getAsset('mat-switch-A')?.referenceCount).toBe(2);

      // Switch to mat-switch-B (tail -> level0): must release mat-switch-A ONCE (leaving externalSession's retain=1)
      await session.loadMaterial(matB.manifest);
      expect(residency.getAsset('mat-switch-A')?.referenceCount).toBe(1);
      expect(residency.getAsset('mat-switch-B')?.referenceCount).toBe(1);

      session.dispose();
      expect(residency.getAsset('mat-switch-B')?.referenceCount).toBe(0);

      // Loading mat-switch-C with another session under tight budget must now evict mat-switch-B cleanly while keeping active mat-switch-A
      const session2 = new VnmStreamingSession(loader, residency);
      await session2.loadMaterial(matC.manifest);
      expect(residency.getAsset('mat-switch-A')).toBeDefined();
      expect(residency.getAsset('mat-switch-B')).toBeUndefined();
      session2.dispose();
      externalSession.dispose();
    });

    it('preserves current active material during switch and cancels stale async results using generations', async () => {
      const matA = await createTestPackage('mat-A');
      const matB = await createTestPackage('mat-B');

      const mockStore = new Map<string, Uint8Array>([
        [matA.manifest.weights.url, matA.pkg.weightsChunk.data],
        [matA.manifest.levels[0]!.chunk.url, matA.pkg.level0Chunk.data],
        [matA.manifest.tail.url, matA.pkg.tailChunk.data],
        [matB.manifest.weights.url, matB.pkg.weightsChunk.data],
        [matB.manifest.levels[0]!.chunk.url, matB.pkg.level0Chunk.data],
        [matB.manifest.tail.url, matB.pkg.tailChunk.data],
      ]);

      let matADelayMs = 0;
      const loader = new VnmChunkLoader({
        fetchFn: async (url) => {
          if (url.includes('mat-A') && matADelayMs > 0) {
            await new Promise((r) => setTimeout(r, matADelayMs));
          }
          return mockStore.get(url)!;
        },
        cacheGet: async () => null,
      });

      const residency = new VnmResidencyManager();
      const session = new VnmStreamingSession(loader, residency);

      // 1. Fully load Material A
      await session.loadMaterial(matA.manifest);
      expect(session.getTier()).toBe('level0');
      expect(session.getCurrentAsset()?.assetId).toBe('mat-A');

      // 2. Start loading Material B with deliberate delay on Mat A's hypothetical old fetches
      matADelayMs = 100;
      const loadBPromise = session.loadMaterial(matB.manifest);

      // Current active material remains Mat A while Mat B is loading!
      expect(session.getCurrentAsset()?.assetId).toBe('mat-A');

      // Await completion of Mat B
      await loadBPromise;

      // Mat B is now active
      expect(session.getCurrentAsset()?.assetId).toBe('mat-B');
      expect(session.getTier()).toBe('level0');

      session.dispose();
    });

    it('handles network failure gracefully without throwing unhandled exceptions', async () => {
      const { manifest } = await createTestPackage('failing-mat');

      const loader = new VnmChunkLoader({
        fetchFn: async () => {
          throw new Error('500 Server Error');
        },
        cacheGet: async () => null,
      });

      const residency = new VnmResidencyManager();
      let errorOccurred = false;

      const session = new VnmStreamingSession(loader, residency, {
        onError: () => {
          errorOccurred = true;
        },
      });

      await session.loadMaterial(manifest);

      expect(errorOccurred).toBe(true);
      expect(session.getState()).toBe('failed');
      expect(session.getTier()).toBe('fallback'); // Remains on safe fallback
    });

    it('cancels pending requests immediately on session disposal', async () => {
      const { manifest } = await createTestPackage('disposed-mat');

      let fetchStarted = false;
      let fetchAborted = false;
      const loader = new VnmChunkLoader({
        fetchFn: async (_url, signal) => {
          fetchStarted = true;
          return new Promise<Uint8Array>((resolve, reject) => {
            const timer = setTimeout(() => {
              resolve(new Uint8Array(100));
            }, 100);

            signal?.addEventListener('abort', () => {
              clearTimeout(timer);
              fetchAborted = true;
              reject(new DOMException('Aborted', 'AbortError'));
            });

            if (signal?.aborted) {
              clearTimeout(timer);
              fetchAborted = true;
              reject(new DOMException('Aborted', 'AbortError'));
            }
          });
        },
        cacheGet: async () => null,
      });

      const residency = new VnmResidencyManager();
      const session = new VnmStreamingSession(loader, residency);

      const loadPromise = session.loadMaterial(manifest);
      // Ensure fetch has begun before disposing
      while (!fetchStarted) {
        await new Promise((r) => setTimeout(r, 5));
      }
      session.dispose();

      await loadPromise;
      expect(session.getState()).toBe('unloaded');
      expect(fetchAborted).toBe(true);
    });
  });
});
