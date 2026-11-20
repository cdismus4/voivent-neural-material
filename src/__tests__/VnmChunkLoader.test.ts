import { describe, it, expect, vi } from 'vitest';
import { VnmChunkLoader } from '../streaming/VnmChunkLoader';
import { generateSyntheticPbrTexture } from '../training/SyntheticTextureGenerator';
import { NeuralMaterialTrainer } from '../training/NeuralMaterialTrainer';
import { exportNeuralMaterial } from '../exporter/VnmExporter';
import type { NeuralMaterialManifest } from '../format/types';

describe('Ticket D1-A: VnmChunkLoader', () => {
  async function createTestPackage() {
    const tex = generateSyntheticPbrTexture(32, 32, 'gradient-waves');
    const trainer = new NeuralMaterialTrainer({
      fineWidth: 16,
      fineHeight: 16,
      coarseWidth: 8,
      coarseHeight: 8,
      iterations: 5,
    });
    const result = trainer.train(tex);
    const pkg = await exportNeuralMaterial(result, tex, {
      assetId: 'test-loader-mat',
    });
    const manifest = JSON.parse(pkg.manifestJson) as NeuralMaterialManifest;
    return { pkg, manifest };
  }

  it('successfully loads and validates weights, latents, and tail chunks from network', async () => {
    const { pkg, manifest } = await createTestPackage();

    const mockNetworkStore = new Map<string, Uint8Array>([
      [manifest.weights.url, pkg.weightsChunk.data],
      [manifest.levels[0]!.chunk.url, pkg.level0Chunk.data],
      [manifest.tail.url, pkg.tailChunk.data],
    ]);

    const fetchFn = vi.fn(async (url: string) => {
      const data = mockNetworkStore.get(url);
      if (!data) throw new Error(`Not found: ${url}`);
      return data;
    });

    const cache = new Map<string, Uint8Array>();
    const loader = new VnmChunkLoader({
      fetchFn,
      cacheGet: async (key) => cache.get(key) ?? null,
      cachePut: async (key, data) => {
        cache.set(key, data);
        return true;
      },
      cacheDelete: async (key) => cache.delete(key),
    });

    // 1. Load weights
    const weightsRes = await loader.loadWeights(manifest);
    expect(weightsRes.weights.w1.length).toBe(24 * 12);
    expect(weightsRes.rawBytes.byteLength).toBe(pkg.weightsChunk.byteLength);
    expect(cache.has(manifest.weights.url)).toBe(true);

    // 2. Load latents
    const latentsRes = await loader.loadLatents(manifest, 0);
    expect(latentsRes.fine.width).toBe(16);
    expect(latentsRes.coarse.width).toBe(8);
    expect(cache.has(manifest.levels[0]!.chunk.url)).toBe(true);

    // 3. Load tail
    const tailRes = await loader.loadTail(manifest);
    expect(tailRes.tail.tailWidth).toBe(16);
    expect(cache.has(manifest.tail.url)).toBe(true);
  });

  it('serves valid chunks from cache without calling network fetch', async () => {
    const { pkg, manifest } = await createTestPackage();

    const fetchFn = vi.fn(async () => {
      throw new Error('Network should not be called on cache hit');
    });

    const cache = new Map<string, Uint8Array>([
      [manifest.weights.url, pkg.weightsChunk.data],
    ]);

    const loader = new VnmChunkLoader({
      fetchFn,
      cacheGet: async (key) => cache.get(key) ?? null,
    });

    const res = await loader.loadWeights(manifest);
    expect(res.weights).toBeDefined();
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it('recovers from corrupted cache: evicts bad cache entry and refetches from network', async () => {
    const { pkg, manifest } = await createTestPackage();

    // Corrupted cache data (hash mismatch)
    const corruptedData = new Uint8Array(pkg.weightsChunk.data.length);
    corruptedData.set(pkg.weightsChunk.data);
    corruptedData[25] = corruptedData[25]! ^ 0xff; // Flip bits

    const cache = new Map<string, Uint8Array>([
      [manifest.weights.url, corruptedData],
    ]);

    let fetchCalled = false;
    const fetchFn = vi.fn(async (_url: string) => {
      fetchCalled = true;
      return pkg.weightsChunk.data;
    });

    const deletedKeys: string[] = [];
    const loader = new VnmChunkLoader({
      fetchFn,
      cacheGet: async (key) => cache.get(key) ?? null,
      cacheDelete: async (key) => {
        deletedKeys.push(key);
        cache.delete(key);
        return true;
      },
      cachePut: async (key, data) => {
        cache.set(key, data);
        return true;
      },
    });

    const res = await loader.loadWeights(manifest);
    expect(res.weights).toBeDefined();
    expect(deletedKeys).toContain(manifest.weights.url);
    expect(fetchCalled).toBe(true);
    // Cache was updated with good data
    expect(cache.get(manifest.weights.url)).toEqual(pkg.weightsChunk.data);
  });

  it('fails gracefully when both cache is corrupted and network fails, without infinite retry', async () => {
    const { pkg, manifest } = await createTestPackage();

    const corruptedData = new Uint8Array(pkg.weightsChunk.data.length);
    corruptedData.fill(0);

    const cache = new Map<string, Uint8Array>([
      [manifest.weights.url, corruptedData],
    ]);

    let networkAttempts = 0;
    const fetchFn = vi.fn(async () => {
      networkAttempts++;
      throw new Error('Network offline 503');
    });

    const loader = new VnmChunkLoader({
      fetchFn,
      cacheGet: async (key) => cache.get(key) ?? null,
      cacheDelete: async (key) => cache.delete(key),
    });

    await expect(loader.loadWeights(manifest)).rejects.toThrow(/Network offline 503/);
    expect(networkAttempts).toBe(1); // Strict single retry, no infinite loop
  });

  it('coalesces simultaneous requests for the same chunk and handles partial cancellation safely', async () => {
    const { pkg, manifest } = await createTestPackage();

    let fetchCalls = 0;
    const fetchFn = vi.fn(async (_url: string, signal?: AbortSignal) => {
      fetchCalls++;
      // Simulate network delay
      await new Promise((resolve) => setTimeout(resolve, 50));
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      return pkg.weightsChunk.data;
    });

    const loader = new VnmChunkLoader({
      fetchFn,
      cacheGet: async () => null,
    });

    const controller1 = new AbortController();
    const controller2 = new AbortController();

    // Launch simultaneous requests from two consumers
    const req1 = loader.loadWeights(manifest, controller1.signal);
    const req2 = loader.loadWeights(manifest, controller2.signal);

    // Cancel consumer 1 mid-flight
    controller1.abort();

    // Consumer 1 should reject with AbortError
    await expect(req1).rejects.toThrow();

    // Consumer 2 MUST still succeed without being cancelled!
    const res2 = await req2;
    expect(res2.weights).toBeDefined();

    // Only 1 underlying network fetch occurred
    expect(fetchCalls).toBe(1);
  });

  it('F-4: rejects chunk when SHA-256 verification is unavailable (e.g. non-secure context without crypto.subtle)', async () => {
    const { pkg, manifest } = await createTestPackage();
    const loader = new VnmChunkLoader({
      fetchFn: async () => pkg.weightsChunk.data,
      cacheGet: async () => pkg.weightsChunk.data,
      cacheDelete: async () => true,
    });

    vi.stubGlobal('crypto', {});
    try {
      await expect(loader.loadWeights(manifest)).rejects.toThrow(/validation failed/i);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('M-4: rejects tail chunk when subheader tailFirstMip mismatches manifest.tailFirstMip', async () => {
    const { pkg, manifest } = await createTestPackage();
    const mismatchedManifest: NeuralMaterialManifest = {
      ...manifest,
      tailFirstMip: manifest.tailFirstMip + 1,
    };
    const loader = new VnmChunkLoader({
      fetchFn: async () => pkg.tailChunk.data,
      cacheGet: async () => null,
    });

    await expect(loader.loadTail(mismatchedManifest)).rejects.toThrow(/Tail first mip mismatch/);
  });
});
