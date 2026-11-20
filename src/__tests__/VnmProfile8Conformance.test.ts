import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import {
  getLatentChannelCount,
  MLP_PROFILES,
  parseManifest,
  type DecoderProfile,
  type LatentFeatureGrid,
  type MlpWeights,
  type NeuralMaterialManifest,
} from '../format/types';
import { validateU8LatentGrid } from '../format/validation';
import {
  parseLatentsChunk,
  parseWeightsChunk,
  serializeLatentsChunk,
  serializeWeightsChunk,
} from '../format/binary';
import {
  decodeNeuralMaterialCpu,
  decodeNeuralMaterialRawCpu,
  type SampleContext,
} from '../reference/CpuReferenceDecoder';
import { createLatentDataTexture } from '../gpu/TslMlpDecoderNode';
import { generateSyntheticPbrTexture } from '../training/SyntheticTextureGenerator';
import { NeuralMaterialTrainer } from '../training/NeuralMaterialTrainer';
import { exportNeuralMaterial } from '../exporter/VnmExporter';
import { auditVnmPackageQuality } from '../training/PbrQualityAudit';
import { VnmChunkLoader } from '../streaming/VnmChunkLoader';

interface GoldenPbrExpected {
  readonly raw: readonly number[];
  readonly baseColor: readonly [number, number, number];
  readonly normal: readonly [number, number, number];
  readonly roughness: number;
  readonly metalness: number;
  readonly ao: number;
}

interface GoldenVectorCase {
  readonly uv: readonly [number, number];
  readonly wrapU: 'repeat' | 'clamp';
  readonly wrapV: 'repeat' | 'clamp';
  readonly normalizedMip: number;
  readonly expected_raw_float: readonly number[];
  readonly expected_pbr_float: GoldenPbrExpected;
  readonly expected_raw_u8: readonly number[];
  readonly expected_pbr_u8: GoldenPbrExpected;
}

interface GoldenVectorFixture {
  readonly meta: {
    readonly profile: 'standard8' | 'quality8';
    readonly fine_size: readonly [number, number];
    readonly coarse_size: readonly [number, number];
    readonly latent_channels_per_grid: number;
    readonly case_count: number;
    readonly gate_max_abs_error: number;
  };
  readonly weights: {
    readonly w1: readonly number[];
    readonly b1: readonly number[];
    readonly w2: readonly number[];
    readonly b2: readonly number[];
    readonly w3: readonly number[];
    readonly b3: readonly number[];
  };
  readonly fine_float: readonly number[];
  readonly coarse_float: readonly number[];
  readonly fine_u8: readonly number[];
  readonly coarse_u8: readonly number[];
  readonly cases: readonly GoldenVectorCase[];
}

const TEST_DIR = dirname(fileURLToPath(import.meta.url));

function loadGoldenFixture(filename: string): GoldenVectorFixture {
  const raw = readFileSync(resolve(TEST_DIR, 'fixtures', filename), 'utf8');
  return JSON.parse(raw) as GoldenVectorFixture;
}

function toMlpWeights(
  profile: DecoderProfile,
  weights: GoldenVectorFixture['weights'],
): MlpWeights {
  return {
    profile,
    w1: Float32Array.from(weights.w1),
    b1: Float32Array.from(weights.b1),
    w2: Float32Array.from(weights.w2),
    b2: Float32Array.from(weights.b2),
    w3: Float32Array.from(weights.w3),
    b3: Float32Array.from(weights.b3),
  };
}

function measurePbrDiff(
  actual: ReturnType<typeof decodeNeuralMaterialCpu>,
  expected: GoldenPbrExpected,
): number {
  let maxErr = 0;
  for (let i = 0; i < 8; i++) {
    maxErr = Math.max(maxErr, Math.abs(actual.raw.values[i]! - expected.raw[i]!));
  }
  for (let i = 0; i < 3; i++) {
    maxErr = Math.max(maxErr, Math.abs(actual.baseColor[i] - expected.baseColor[i]));
    maxErr = Math.max(maxErr, Math.abs(actual.normal[i] - expected.normal[i]));
  }
  maxErr = Math.max(maxErr, Math.abs(actual.roughness - expected.roughness));
  maxErr = Math.max(maxErr, Math.abs(actual.metalness - expected.metalness));
  maxErr = Math.max(maxErr, Math.abs(actual.ao - expected.ao));
  return maxErr;
}

describe('VNM1 Format Revision: standard8 & quality8 Profiles', () => {
  it('registers standard8 and quality8 in MLP_PROFILES and derives latent channels correctly', () => {
    expect(MLP_PROFILES.standard8).toEqual({
      profile: 'standard8',
      inputDim: 20,
      hidden1: 24,
      hidden2: 24,
      outputDim: 8,
      totalWeights: 1304,
      macsPerEvaluation: 1248,
    });
    expect(MLP_PROFILES.quality8).toEqual({
      profile: 'quality8',
      inputDim: 20,
      hidden1: 32,
      hidden2: 32,
      outputDim: 8,
      totalWeights: 1992,
      macsPerEvaluation: 1920,
    });

    expect(getLatentChannelCount('small')).toBe(4);
    expect(getLatentChannelCount('standard')).toBe(4);
    expect(getLatentChannelCount('quality')).toBe(4);
    expect(getLatentChannelCount('standard8')).toBe(8);
    expect(getLatentChannelCount('quality8')).toBe(8);
  });

  it('parses VNM1 v1 manifests with standard8/quality8 and rejects unknown profiles', () => {
    const sha = 'a'.repeat(64);
    const makeRawManifest = (profile: string) => ({
      format: 'VNM1',
      version: 1,
      assetId: `mat-${profile}`,
      sourceSize: [64, 64],
      profile,
      quantization: 'u8',
      weightStoragePrecision: 'f32',
      computePrecision: 'f32',
      channels: 'base3-normal2-rough-metal-ao',
      baseColorSpace: 'linear-srgb',
      normalConvention: 'tangent-positive-y-positive-z',
      wrapU: 'repeat',
      wrapV: 'repeat',
      weights: { url: 'weights.vnm', byteLength: 16, sha256: sha },
      tailFirstMip: 3,
      tail: { url: 'tail.vnm', byteLength: 16, sha256: sha },
      levels: [
        {
          sourceMip: 0,
          sourceSize: [64, 64],
          fineSize: [64, 64],
          coarseSize: [32, 32],
          chunk: { url: 'level0.vnm', byteLength: 16, sha256: sha },
        },
      ],
    });

    for (const profile of ['standard8', 'quality8'] as const) {
      const manifest = parseManifest(makeRawManifest(profile));
      expect(manifest.profile).toBe(profile);
    }

    expect(() => parseManifest(makeRawManifest('ultra8'))).toThrow(/Invalid profile/);
  });

  for (const [profile, filename] of [
    ['standard8', 'golden_vectors_standard8.json'],
    ['quality8', 'golden_vectors_quality8.json'],
  ] as const) {
    it(`reproduces all 286 golden cases (${profile}) for float, u8, binary roundtrip, and PBR within 1e-4`, () => {
      const fixture = loadGoldenFixture(filename);
      expect(fixture.meta.profile).toBe(profile);
      expect(fixture.meta.latent_channels_per_grid).toBe(8);
      expect(fixture.meta.case_count).toBe(286);
      expect(fixture.cases).toHaveLength(286);

      const gate = fixture.meta.gate_max_abs_error;
      expect(gate).toBe(1e-4);

      const [fineW, fineH] = fixture.meta.fine_size;
      const [coarseW, coarseH] = fixture.meta.coarse_size;
      const weights = toMlpWeights(profile, fixture.weights);

      const fineFloatGrid: LatentFeatureGrid = {
        width: fineW,
        height: fineH,
        channels: 8,
        data: Float32Array.from(fixture.fine_float),
        isNormalizedU8: false,
      };
      const coarseFloatGrid: LatentFeatureGrid = {
        width: coarseW,
        height: coarseH,
        channels: 8,
        data: Float32Array.from(fixture.coarse_float),
        isNormalizedU8: false,
      };

      const fineU8Grid: LatentFeatureGrid = {
        width: fineW,
        height: fineH,
        channels: 8,
        data: Uint8Array.from(fixture.fine_u8),
        isNormalizedU8: true,
      };
      const coarseU8Grid: LatentFeatureGrid = {
        width: coarseW,
        height: coarseH,
        channels: 8,
        data: Uint8Array.from(fixture.coarse_u8),
        isNormalizedU8: true,
      };

      // Binary roundtrip for weights and latents (VNM1 format v1, reserved bytes 10..15 == 0)
      const weightsChunk = serializeWeightsChunk(weights);
      const parsedWeights = parseWeightsChunk(weightsChunk, profile);

      const latentsChunk = serializeLatentsChunk(fineU8Grid, coarseU8Grid, profile);
      // Verify file header (16B) + subheader bytes 10..15 (file bytes 26..31) are strictly 0
      for (let b = 26; b < 32; b++) {
        expect(latentsChunk[b]).toBe(0);
      }
      const parsedLatents = parseLatentsChunk(latentsChunk, profile);
      expect(parsedLatents.fine.channels).toBe(8);
      expect(parsedLatents.coarse.channels).toBe(8);
      expect(validateU8LatentGrid(parsedLatents.fine, 8)).toBe(8);
      expect(validateU8LatentGrid(parsedLatents.coarse, 8)).toBe(8);

      let maxRawFloatErr = 0;
      let maxPbrFloatErr = 0;
      let maxRawU8Err = 0;
      let maxPbrU8Err = 0;

      for (let idx = 0; idx < fixture.cases.length; idx++) {
        const tc = fixture.cases[idx]!;
        const ctx: SampleContext = {
          u: tc.uv[0],
          v: tc.uv[1],
          wrapU: tc.wrapU,
          wrapV: tc.wrapV,
          normalizedMip: tc.normalizedMip,
        };

        // Float path
        const rawFloat = decodeNeuralMaterialRawCpu(fineFloatGrid, coarseFloatGrid, weights, ctx);
        for (let c = 0; c < 8; c++) {
          maxRawFloatErr = Math.max(maxRawFloatErr, Math.abs(rawFloat[c]! - tc.expected_raw_float[c]!));
        }
        const pbrFloat = decodeNeuralMaterialCpu(fineFloatGrid, coarseFloatGrid, weights, ctx);
        maxPbrFloatErr = Math.max(maxPbrFloatErr, measurePbrDiff(pbrFloat, tc.expected_pbr_float));

        // U8 path via binary roundtrip
        const rawU8 = decodeNeuralMaterialRawCpu(
          parsedLatents.fine,
          parsedLatents.coarse,
          parsedWeights,
          ctx,
        );
        for (let c = 0; c < 8; c++) {
          maxRawU8Err = Math.max(maxRawU8Err, Math.abs(rawU8[c]! - tc.expected_raw_u8[c]!));
        }
        const pbrU8 = decodeNeuralMaterialCpu(
          parsedLatents.fine,
          parsedLatents.coarse,
          parsedWeights,
          ctx,
        );
        maxPbrU8Err = Math.max(maxPbrU8Err, measurePbrDiff(pbrU8, tc.expected_pbr_u8));
      }

      console.info(
        `[VnmProfile8Conformance:${profile}] 286 cases maxAbsError -> rawFloat=${maxRawFloatErr.toExponential(3)}, pbrFloat=${maxPbrFloatErr.toExponential(3)}, rawU8=${maxRawU8Err.toExponential(3)}, pbrU8=${maxPbrU8Err.toExponential(3)}`,
      );

      expect(maxRawFloatErr).toBeLessThanOrEqual(gate);
      expect(maxPbrFloatErr).toBeLessThanOrEqual(gate);
      expect(maxRawU8Err).toBeLessThanOrEqual(gate);
      expect(maxPbrU8Err).toBeLessThanOrEqual(gate);
    });
  }

  it('rejects non-zero subheader reserved bytes 10..15 for 8ch latent chunks', () => {
    const fine: LatentFeatureGrid = {
      width: 4,
      height: 4,
      channels: 8,
      data: new Uint8Array(4 * 4 * 8),
      isNormalizedU8: true,
    };
    const coarse: LatentFeatureGrid = {
      width: 2,
      height: 2,
      channels: 8,
      data: new Uint8Array(2 * 2 * 8),
      isNormalizedU8: true,
    };

    for (const profile of ['standard8', 'quality8'] as const) {
      const chunk = serializeLatentsChunk(fine, coarse, profile);
      for (let offset = 26; offset < 32; offset++) {
        const corrupted = new Uint8Array(chunk);
        corrupted[offset] = 8;
        expect(() => parseLatentsChunk(corrupted, profile)).toThrow(/reserved byte/);
      }
    }
  });

  it('rejects 8ch latent grids on the 4ch GPU DataTexture helper', () => {
    const grid8: LatentFeatureGrid = {
      width: 4,
      height: 4,
      channels: 8,
      data: new Uint8Array(4 * 4 * 8),
      isNormalizedU8: true,
    };
    expect(() => createLatentDataTexture(grid8)).toThrow(/channel count mismatch/);

    const grid4: LatentFeatureGrid = {
      width: 4,
      height: 4,
      channels: 4,
      data: new Uint8Array(4 * 4 * 4),
      isNormalizedU8: true,
    };
    expect(() => createLatentDataTexture(grid4)).not.toThrow();
  });

  it('supports train -> export -> VnmChunkLoader -> audit -> CPU decode end-to-end for standard8 and quality8', async () => {
    const source = generateSyntheticPbrTexture(16, 16, 'gradient-waves');

    for (const profile of ['standard8', 'quality8'] as const) {
      const trainer = new NeuralMaterialTrainer({
        profile,
        fineWidth: 8,
        fineHeight: 8,
        coarseWidth: 4,
        coarseHeight: 4,
        batchSize: 32,
        iterations: 4,
      });
      const training = trainer.train(source);
      expect(training.weights.profile).toBe(profile);
      expect(training.weights.w1.length).toBe(
        MLP_PROFILES[profile].hidden1 * MLP_PROFILES[profile].inputDim,
      );
      expect(training.fineGridFloat.length).toBe(8 * 8 * 8);
      expect(training.coarseGridFloat.length).toBe(4 * 4 * 8);

      const pkg = await exportNeuralMaterial(training, source, {
        assetId: `e2e-${profile}`,
      });
      expect(pkg.manifest.profile).toBe(profile);
      expect(pkg.level0Chunk.byteLength).toBe(16 + 16 + (8 * 8 * 8 + 4 * 4 * 8));

      const manifest = JSON.parse(pkg.manifestJson) as NeuralMaterialManifest;
      const store = new Map<string, Uint8Array>([
        [manifest.weights.url, pkg.weightsChunk.data],
        [manifest.levels[0]!.chunk.url, pkg.level0Chunk.data],
      ]);
      const cache = new Map<string, Uint8Array>();
      const loader = new VnmChunkLoader({
        fetchFn: vi.fn(async (url: string) => {
          const data = store.get(url);
          if (!data) throw new Error(`Missing url: ${url}`);
          return data;
        }),
        cacheGet: async (key) => cache.get(key) ?? null,
        cachePut: async (key, data) => {
          cache.set(key, data);
          return true;
        },
        cacheDelete: async (key) => cache.delete(key),
      });

      const loadedWeights = await loader.loadWeights(manifest);
      const loadedLatents = await loader.loadLatents(manifest, 0);
      expect(loadedWeights.weights.profile).toBe(profile);
      expect(loadedLatents.fine.channels).toBe(8);
      expect(loadedLatents.coarse.channels).toBe(8);

      const audit = auditVnmPackageQuality(source, pkg, 8);
      expect(audit.sampleCount).toBe(64);
      expect(Number.isFinite(audit.overallMae)).toBe(true);

      const decoded = decodeNeuralMaterialCpu(
        loadedLatents.fine,
        loadedLatents.coarse,
        loadedWeights.weights,
        { u: 0.25, v: 0.75, wrapU: 'repeat', wrapV: 'repeat', normalizedMip: 0.5 },
      );
      expect(decoded.raw.values).toHaveLength(8);
      for (const ch of decoded.raw.values) {
        expect(Number.isFinite(ch)).toBe(true);
      }
    }
  });
});
