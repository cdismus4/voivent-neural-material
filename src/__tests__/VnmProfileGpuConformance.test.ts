import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import {
    BoxGeometry,
    FloatType,
    LinearFilter,
    Mesh,
    MeshStandardNodeMaterial,
    NoColorSpace,
    PerspectiveCamera,
    RepeatWrapping,
    RGBAFormat,
    StorageBufferAttribute,
    UnsignedByteType,
    WebGPURenderer,
    WGSLNodeBuilder,
} from 'three/webgpu';
import { uv } from 'three/tsl';
import {
    getLatentChannelCount,
    MLP_PROFILES,
    type DecoderProfile,
    type LatentFeatureGrid,
    type MlpWeights,
} from '../format/types';
import { parseLatentsChunk, parseWeightsChunk, serializeLatentsChunk, serializeWeightsChunk } from '../format/binary';
import {
    decodeNeuralMaterialCpu,
    decodeNeuralMaterialRawCpu,
    type SampleContext,
} from '../reference/CpuReferenceDecoder';
import {
    attachNeuralPbrNodes,
    createLatentDataTexture,
    createLatentDataTexturePair,
    createLatentTextureSource,
    createNeuralMaterialTslNodes,
    createWeightsStorageBuffer,
} from '../gpu/TslMlpDecoderNode';
import {
    createVerificationLatentGrids,
    createVerificationWeights,
    runGpuNeuralMaterialVerification,
} from '../gpu/GpuReadbackVerification';

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
        readonly profile: DecoderProfile;
        readonly fine_size: readonly [number, number];
        readonly coarse_size: readonly [number, number];
        readonly latent_channels_per_grid?: number;
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
const ALL_PROFILES: readonly DecoderProfile[] = ['small', 'standard', 'quality', 'standard8', 'quality8'];

function loadGoldenFixture(filename: string): GoldenVectorFixture {
    const raw = readFileSync(resolve(TEST_DIR, 'fixtures', filename), 'utf8');
    return JSON.parse(raw) as GoldenVectorFixture;
}

function toMlpWeights(profile: DecoderProfile, weights: GoldenVectorFixture['weights']): MlpWeights {
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
    for (let c = 0; c < 3; c++) {
        maxErr = Math.max(maxErr, Math.abs(actual.baseColor[c]! - expected.baseColor[c]!));
        maxErr = Math.max(maxErr, Math.abs(actual.normal[c]! - expected.normal[c]!));
    }
    maxErr = Math.max(maxErr, Math.abs(actual.roughness - expected.roughness));
    maxErr = Math.max(maxErr, Math.abs(actual.metalness - expected.metalness));
    maxErr = Math.max(maxErr, Math.abs(actual.ao - expected.ao));
    return maxErr;
}

interface WGSLNodeBuilderWithShader {
    camera: PerspectiveCamera;
    fragmentShader: string;
    build(): void;
}

function createMockWgslRenderer(): WebGPURenderer {
    const renderer = new WebGPURenderer();
    (renderer as unknown as { _initialized: boolean })._initialized = true;
    (renderer as unknown as { backend: unknown }).backend = {
        hasFeature: () => true,
        utils: {
            getTextureSampleData: () => ({ primarySamples: 1 }),
        },
    };
    (renderer as unknown as { hasFeature: () => boolean }).hasFeature = () => true;
    return renderer;
}

function withMockDocumentCanvas<T>(fn: () => T): T {
    const g = globalThis as unknown as { document?: unknown };
    const origDoc = g.document;
    const mockCanvas = {
        width: 800,
        height: 600,
        style: {},
        getContext: () => null,
        addEventListener: () => {},
        removeEventListener: () => {},
    };
    g.document = {
        createElement: () => mockCanvas,
        createElementNS: () => mockCanvas,
    };
    try {
        return fn();
    } finally {
        g.document = origDoc;
    }
}

describe('VNM1 P2: Multi-Profile GPU TSL Decoder & Golden Conformance', () => {
    it.each([
        { profile: 'standard' as const, file: 'golden_vectors_standard.json' },
        { profile: 'quality' as const, file: 'golden_vectors_quality.json' },
    ])('reproduces all 286 golden cases ($profile) for float, u8, binary roundtrip, and PBR within 1e-4', ({
        profile,
        file,
    }) => {
        const fixture = loadGoldenFixture(file);
        expect(fixture.meta.profile).toBe(profile);
        expect(fixture.meta.case_count).toBe(286);
        expect(fixture.cases).toHaveLength(286);

        const weights = toMlpWeights(profile, fixture.weights);
        const channels = fixture.meta.latent_channels_per_grid ?? getLatentChannelCount(profile);
        const [fw, fh] = fixture.meta.fine_size;
        const [cw, ch] = fixture.meta.coarse_size;

        const fineFloat: LatentFeatureGrid = {
            width: fw,
            height: fh,
            channels,
            data: Float32Array.from(fixture.fine_float),
            isNormalizedU8: false,
        };
        const coarseFloat: LatentFeatureGrid = {
            width: cw,
            height: ch,
            channels,
            data: Float32Array.from(fixture.coarse_float),
            isNormalizedU8: false,
        };
        const fineU8: LatentFeatureGrid = {
            width: fw,
            height: fh,
            channels,
            data: Uint8Array.from(fixture.fine_u8),
            isNormalizedU8: true,
        };
        const coarseU8: LatentFeatureGrid = {
            width: cw,
            height: ch,
            channels,
            data: Uint8Array.from(fixture.coarse_u8),
            isNormalizedU8: true,
        };

        const parsedWeights = parseWeightsChunk(serializeWeightsChunk(weights), profile);
        const parsedLatents = parseLatentsChunk(serializeLatentsChunk(fineU8, coarseU8, profile), profile);

        let maxRawFloatErr = 0;
        let maxPbrFloatErr = 0;
        let maxRawU8Err = 0;
        let maxPbrU8Err = 0;

        for (const tc of fixture.cases) {
            const ctx: SampleContext = {
                u: tc.uv[0],
                v: tc.uv[1],
                wrapU: tc.wrapU,
                wrapV: tc.wrapV,
                normalizedMip: tc.normalizedMip,
            };

            const actualRawFloat = decodeNeuralMaterialRawCpu(fineFloat, coarseFloat, weights, ctx);
            const actualPbrFloat = decodeNeuralMaterialCpu(fineFloat, coarseFloat, weights, ctx);
            const actualRawU8 = decodeNeuralMaterialRawCpu(parsedLatents.fine, parsedLatents.coarse, parsedWeights, ctx);
            const actualPbrU8 = decodeNeuralMaterialCpu(parsedLatents.fine, parsedLatents.coarse, parsedWeights, ctx);

            for (let i = 0; i < 8; i++) {
                maxRawFloatErr = Math.max(maxRawFloatErr, Math.abs(actualRawFloat[i]! - tc.expected_raw_float[i]!));
                maxRawU8Err = Math.max(maxRawU8Err, Math.abs(actualRawU8[i]! - tc.expected_raw_u8[i]!));
            }
            maxPbrFloatErr = Math.max(maxPbrFloatErr, measurePbrDiff(actualPbrFloat, tc.expected_pbr_float));
            maxPbrU8Err = Math.max(maxPbrU8Err, measurePbrDiff(actualPbrU8, tc.expected_pbr_u8));
        }

        expect(maxRawFloatErr).toBeLessThan(1e-4);
        expect(maxPbrFloatErr).toBeLessThan(1e-4);
        expect(maxRawU8Err).toBeLessThan(1e-4);
        expect(maxPbrU8Err).toBeLessThan(1e-4);
    });

    it.each(ALL_PROFILES)('packs StorageBuffer offsets cleanly across profile %s', (profile) => {
        const spec = MLP_PROFILES[profile];
        const weights = createVerificationWeights(314, profile);
        const attr = createWeightsStorageBuffer(weights);
        const buf = attr.array as Float32Array;

        const inVec4s = spec.inputDim / 4;
        const h1Vec4s = spec.hidden1 / 4;
        const h2Vec4s = spec.hidden2 / 4;
        const expectedTotalVec4s =
            spec.hidden1 * inVec4s +
            h1Vec4s +
            spec.hidden2 * h1Vec4s +
            h2Vec4s +
            spec.outputDim * h2Vec4s +
            spec.outputDim / 4;

        expect(attr.itemSize).toBe(4);
        expect(attr.count).toBe(expectedTotalVec4s);
        expect(buf.length).toBe(expectedTotalVec4s * 4);

        const w1Start = 0;
        const b1Start = (spec.hidden1 * inVec4s) * 4;
        const w2Start = b1Start + spec.hidden1;
        const b2Start = w2Start + spec.hidden2 * spec.hidden1;
        const w3Start = b2Start + spec.hidden2;
        const b3Start = w3Start + spec.outputDim * spec.hidden2;

        expect(Array.from(buf.subarray(w1Start, b1Start))).toEqual(Array.from(weights.w1));
        expect(Array.from(buf.subarray(b1Start, w2Start))).toEqual(Array.from(weights.b1));
        expect(Array.from(buf.subarray(w2Start, b2Start))).toEqual(Array.from(weights.w2));
        expect(Array.from(buf.subarray(b2Start, w3Start))).toEqual(Array.from(weights.b2));
        expect(Array.from(buf.subarray(w3Start, b3Start))).toEqual(Array.from(weights.w3));
        expect(Array.from(buf.subarray(b3Start, b3Start + spec.outputDim))).toEqual(Array.from(weights.b3));
    });

    it('splits 8ch latent grids into RGBA8 texture pairs accurately and guards mismatched channel counts', () => {
        const { fine: fine8 } = createVerificationLatentGrids(4, 3, 2, 2, 99, 8);
        const [tex0, tex1] = createLatentDataTexturePair(fine8, 'repeat', 'clamp');

        expect(tex0.type).toBe(UnsignedByteType);
        expect(tex0.format).toBe(RGBAFormat);
        expect(tex0.minFilter).toBe(LinearFilter);
        expect(tex0.magFilter).toBe(LinearFilter);
        expect(tex0.colorSpace).toBe(NoColorSpace);
        expect(tex0.generateMipmaps).toBe(false);
        expect(tex0.flipY).toBe(false);
        expect(tex0.wrapS).toBe(RepeatWrapping);

        const d0 = tex0.image.data as Uint8Array;
        const d1 = tex1.image.data as Uint8Array;
        expect(d0.length).toBe(4 * 3 * 4);
        expect(d1.length).toBe(4 * 3 * 4);

        for (let px = 0; px < 4 * 3; px++) {
            for (let c = 0; c < 4; c++) {
                expect(d0[px * 4 + c]).toBe(fine8.data[px * 8 + c]);
                expect(d1[px * 4 + c]).toBe(fine8.data[px * 8 + 4 + c]);
            }
        }

        // Source factory returns single texture for 4ch and tuple for 8ch
        const { fine: fine4 } = createVerificationLatentGrids(4, 3, 2, 2, 99, 4);
        expect(Array.isArray(createLatentTextureSource(fine4, 'quality'))).toBe(false);
        expect(Array.isArray(createLatentTextureSource(fine8, 'standard8'))).toBe(true);
        expect(Array.isArray(createLatentTextureSource(fine8, 'quality8'))).toBe(true);

        // Strict boundary guards on texture constructors
        expect(() => createLatentDataTexture(fine8)).toThrow(/byteLength|channels/);
        expect(() => createLatentDataTexturePair(fine4)).toThrow(/byteLength|channels/);
        expect(() => createLatentTextureSource(fine4, 'standard8')).toThrow(/byteLength|channels/);
        expect(() => createLatentTextureSource(fine8, 'standard')).toThrow(/byteLength|channels/);
    });

    it('enforces strict boundary guards on createNeuralMaterialTslNodes profile/texture/weight mismatches', () => {
        const { fine: f4, coarse: c4 } = createVerificationLatentGrids(4, 4, 2, 2, 11, 4);
        const { fine: f8, coarse: c8 } = createVerificationLatentGrids(4, 4, 2, 2, 22, 8);
        const singleFine = createLatentDataTexture(f4);
        const singleCoarse = createLatentDataTexture(c4);
        const pairFine = createLatentDataTexturePair(f8);
        const pairCoarse = createLatentDataTexturePair(c8);
        const wQuality = createVerificationWeights(1, 'quality');
        const wStd8 = createVerificationWeights(2, 'standard8');
        const badBuf = new StorageBufferAttribute(new Float32Array(16), 4);
        const uvNode = uv();

        // Unknown profile rejected explicitly
        expect(() =>
            createNeuralMaterialTslNodes(singleFine, singleCoarse, badBuf, { uv: uvNode }, {
                profile: 'ultra99' as unknown as DecoderProfile,
                fineSize: [4, 4],
                coarseSize: [2, 2],
            }),
        ).toThrow(/Unsupported profile/);

        // 8ch profile with single texture rejected explicitly
        expect(() =>
            createNeuralMaterialTslNodes(singleFine, singleCoarse, wStd8, { uv: uvNode }, {
                profile: 'standard8',
                fineSize: [4, 4],
                coarseSize: [2, 2],
            }),
        ).toThrow(/requires 2 RGBA8 texture/);

        // 4ch profile with pair texture rejected explicitly
        expect(() =>
            createNeuralMaterialTslNodes(pairFine, pairCoarse, wQuality, { uv: uvNode }, {
                profile: 'quality',
                fineSize: [4, 4],
                coarseSize: [2, 2],
            }),
        ).toThrow(/requires 1 RGBA8 texture/);

        // Mismatched weights profile rejected explicitly
        expect(() =>
            createNeuralMaterialTslNodes(pairFine, pairCoarse, wQuality, { uv: uvNode }, {
                profile: 'standard8',
                fineSize: [4, 4],
                coarseSize: [2, 2],
            }),
        ).toThrow(/Profile mismatch/);

        // Mismatched StorageBufferAttribute size rejected explicitly
        expect(() =>
            createNeuralMaterialTslNodes(pairFine, pairCoarse, badBuf, { uv: uvNode }, {
                profile: 'quality8',
                fineSize: [4, 4],
                coarseSize: [2, 2],
            }),
        ).toThrow(/StorageBufferAttribute must have/);
    });

    it.each(ALL_PROFILES)(
        'compiles real WGSL shader cleanly with weight buffer invariance for profile %s',
        (profile) => {
            const chCount = getLatentChannelCount(profile);
            const { fine, coarse } = createVerificationLatentGrids(8, 8, 4, 4, 100, chCount === 8 ? 8 : 4);
            const fineSource = createLatentTextureSource(fine, profile);
            const coarseSource = createLatentTextureSource(coarse, profile);

            const weightsA = createVerificationWeights(111, profile);
            const weightsB = createVerificationWeights(999, profile);
            const camera = new PerspectiveCamera(45, 1, 0.1, 100);
            const uvNode = uv();

            withMockDocumentCanvas(() => {
                const matA = new MeshStandardNodeMaterial();
                const nodesA = createNeuralMaterialTslNodes(fineSource, coarseSource, weightsA, { uv: uvNode }, {
                    profile,
                    fineSize: [8, 8],
                    coarseSize: [4, 4],
                });
                attachNeuralPbrNodes(matA, nodesA);

                const meshA = new Mesh(new BoxGeometry(), matA);
                const builderA = new WGSLNodeBuilder(meshA, createMockWgslRenderer()) as unknown as WGSLNodeBuilderWithShader;
                builderA.camera = camera;
                builderA.build();

                const matB = new MeshStandardNodeMaterial();
                const nodesB = createNeuralMaterialTslNodes(fineSource, coarseSource, weightsB, { uv: uvNode }, {
                    profile,
                    fineSize: [8, 8],
                    coarseSize: [4, 4],
                });
                attachNeuralPbrNodes(matB, nodesB);

                const meshB = new Mesh(new BoxGeometry(), matB);
                const builderB = new WGSLNodeBuilder(meshB, createMockWgslRenderer()) as unknown as WGSLNodeBuilderWithShader;
                builderB.camera = camera;
                builderB.build();

                expect(builderA.fragmentShader).toContain('var<storage, read>');
                const normalizeWgsl = (code: string) => code.replace(/NodeBuffer_\d+/g, 'NodeBuffer_WEIGHTS');
                expect(normalizeWgsl(builderA.fragmentShader)).toBe(normalizeWgsl(builderB.fragmentShader));

                // 8ch binds 4 RGBA textures (2 fine + 2 coarse); 4ch binds 2 RGBA textures (1 fine + 1 coarse)
                const texDeclCount = (builderA.fragmentShader.match(/texture_2d<f32>/g) ?? []).length;
                expect(texDeclCount).toBe(chCount === 8 ? 4 : 2);
            });
        },
    );

    it.each(['quality', 'standard8', 'quality8'] as const)(
        'runs multi-profile verification harness cleanly and disposes all textures for %s',
        async (profile) => {
            const channels: 4 | 8 = getLatentChannelCount(profile) === 8 ? 8 : 4;
            const weights = createVerificationWeights(12345, profile);
            const { fine, coarse } = createVerificationLatentGrids(1, 1, 1, 1, 42, channels);
            const res = 4;

            const fakeReadPixels = (idx: 0 | 1): Float32Array => {
                const rgbaFloat = new Float32Array(res * res * 4);
                for (let row = 0; row < res; row++) {
                    for (let col = 0; col < res; col++) {
                        const u = (col + 0.5) / res;
                        const v = (row + 0.5) / res;
                        const expected = decodeNeuralMaterialRawCpu(fine, coarse, weights, {
                            u,
                            v,
                            wrapU: 'repeat',
                            wrapV: 'repeat',
                        });
                        const off = (row * res + col) * 4;
                        const chOffset = idx === 0 ? 0 : 4;
                        for (let c = 0; c < 4; c++) {
                            rgbaFloat[off + c] = expected[chOffset + c]!;
                        }
                    }
                }
                return rgbaFloat;
            };

            let renderPass = 0;
            const mockRenderer = {
                backend: {
                    isWebGPUBackend: true,
                    adapter: {
                        info: { vendor: 'mock', architecture: 'test', device: `Mock ${profile}`, description: `Mock ${profile}` },
                    },
                },
                init: vi.fn(async () => {}),
                getActiveCubeFace: () => 0,
                getActiveMipmapLevel: () => 0,
                getRenderTarget: () => null,
                setRenderTarget: vi.fn(),
                render: vi.fn(),
                readRenderTargetPixelsAsync: vi.fn(async (rt: { texture?: { type?: number; format?: number } }) => {
                    expect(rt.texture?.type).toBe(FloatType);
                    expect(rt.texture?.format).toBe(RGBAFormat);
                    const idx: 0 | 1 = (renderPass++ % 2) === 0 ? 0 : 1;
                    return fakeReadPixels(idx);
                }),
            };

            const report = await runGpuNeuralMaterialVerification(mockRenderer as unknown as WebGPURenderer, {
                profile,
                filterCaseIds: ['minimal-1x1-grid'],
                resolution: res,
                tolerance: 1e-4,
            });

            expect(report.profile).toBe(profile);
            expect(report.passed).toBe(true);
            expect(report.maxErrorAcrossAllTests).toBeLessThan(1e-4);
            expect(report.testCases).toHaveLength(1);
            expect(report.markdownReport).toContain(`\`${profile}\``);
        },
    );
});
