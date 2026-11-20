/**
 * Voivent Neural Material (VNM1) Format & Specification
 * Clean-room neural texture compression and procedural PBR representation.
 */

import { validateManifestLevels } from './validation';

export type DecoderProfile = 'small' | 'standard' | 'quality' | 'standard8' | 'quality8';
export type Quantization = 'u8' | 'u4';
export type DecodeMode = 'direct' | 'cached' | 'tail';
export type Precision = 'f32' | 'f16';

export interface ChunkRef {
    readonly url: string;
    readonly sha256: string;
    readonly byteLength: number;
}

export interface LatentLevel {
    readonly sourceMip: number;
    readonly sourceSize: readonly [number, number];
    readonly fineSize: readonly [number, number];
    readonly coarseSize: readonly [number, number];
    readonly chunk: ChunkRef;
}

export interface NeuralMaterialManifest {
    readonly format: 'VNM1';
    readonly version: 1;
    readonly assetId: string;
    readonly sourceSize: readonly [number, number];
    readonly profile: DecoderProfile;
    readonly quantization: Quantization;
    /** Storage format of weights.bin */
    readonly weightStoragePrecision: Precision;
    /** Target compute precision during shading */
    readonly computePrecision: Precision;
    readonly channels: 'base3-normal2-rough-metal-ao';
    readonly baseColorSpace: 'linear-srgb';
    readonly normalConvention: 'tangent-positive-y-positive-z';
    readonly wrapU: 'repeat' | 'clamp';
    readonly wrapV: 'repeat' | 'clamp';
    readonly weights: ChunkRef;
    readonly levels: readonly LatentLevel[];
    readonly tailFirstMip: number;
    readonly mipMax?: number;
    readonly tail: ChunkRef;
    readonly fallbackMaterialUrl?: string;
}

/**
 * Resolves the normalizedMip denominator (mipMax) per SPEC §6.2 (Option C):
 * explicit `manifest.mipMax` if present, otherwise `max(1, tailFirstMip - 1)`.
 */
export function resolveManifestMipMax(
    manifest: Pick<NeuralMaterialManifest, 'mipMax' | 'tailFirstMip'>,
): number {
    if (manifest.mipMax !== undefined) {
        if (!Number.isFinite(manifest.mipMax) || manifest.mipMax <= 0) {
            throw new Error(`Invalid mipMax: '${String(manifest.mipMax)}': must be a positive finite number`);
        }
        return manifest.mipMax;
    }
    return Math.max(1, manifest.tailFirstMip - 1);
}

export interface MlpProfileSpec {
    readonly profile: DecoderProfile;
    readonly inputDim: number;
    readonly hidden1: number;
    readonly hidden2: number;
    readonly outputDim: number;
    readonly totalWeights: number;
    readonly macsPerEvaluation: number;
}

export const MLP_PROFILES: Record<DecoderProfile, MlpProfileSpec> = {
    small: {
        profile: 'small',
        inputDim: 12,
        hidden1: 16,
        hidden2: 16,
        outputDim: 8,
        totalWeights: (16 * 12 + 16) + (16 * 16 + 16) + (8 * 16 + 8), // 208 + 272 + 136 = 616
        macsPerEvaluation: 576,
    },
    standard: {
        profile: 'standard',
        inputDim: 12,
        hidden1: 24,
        hidden2: 24,
        outputDim: 8,
        totalWeights: (24 * 12 + 24) + (24 * 24 + 24) + (8 * 24 + 8), // 312 + 600 + 200 = 1112
        macsPerEvaluation: 1056,
    },
    quality: {
        profile: 'quality',
        inputDim: 12,
        hidden1: 32,
        hidden2: 32,
        outputDim: 8,
        totalWeights: (32 * 12 + 32) + (32 * 32 + 32) + (8 * 32 + 8), // 416 + 1056 + 264 = 1736
        macsPerEvaluation: 1664,
    },
    // Added by the 2026-11-18 format revision (profile-registry
    // extension; format name and version unchanged). The 8-series
    // profiles use 8+8 latent channels, so inputDim is 20
    // (8 fine + 8 coarse + 2 local + 1 mip + 1 const); the latent
    // channel count per grid derives as (inputDim - 4) / 2.
    standard8: {
        profile: 'standard8',
        inputDim: 20,
        hidden1: 24,
        hidden2: 24,
        outputDim: 8,
        totalWeights: (24 * 20 + 24) + (24 * 24 + 24) + (8 * 24 + 8), // 504 + 600 + 200 = 1304
        macsPerEvaluation: 1248,
    },
    quality8: {
        profile: 'quality8',
        inputDim: 20,
        hidden1: 32,
        hidden2: 32,
        outputDim: 8,
        totalWeights: (32 * 20 + 32) + (32 * 32 + 32) + (8 * 32 + 8), // 672 + 1056 + 264 = 1992
        macsPerEvaluation: 1920,
    },
};

/**
 * Derives the per-grid latent channel count from a VNM1 profile: (inputDim - 4) / 2.
 */
export function getLatentChannelCount(profile: DecoderProfile): number {
    const spec = MLP_PROFILES[profile];
    if (!spec) {
        throw new Error(`Unsupported profile: '${String(profile)}'`);
    }
    return (spec.inputDim - 4) / 2;
}

export interface MlpWeights {
    readonly profile: DecoderProfile;
    readonly w1: Float32Array; // hidden1 x inputDim
    readonly b1: Float32Array; // hidden1
    readonly w2: Float32Array; // hidden2 x hidden1
    readonly b2: Float32Array; // hidden2
    readonly w3: Float32Array; // outputDim x hidden2
    readonly b3: Float32Array; // outputDim
}

export interface RawMlpOutputs {
    readonly values: readonly [number, number, number, number, number, number, number, number];
}

export interface DecodedPbrChannels {
    readonly raw: RawMlpOutputs;
    readonly baseColor: readonly [number, number, number]; // Linear RGB, 0..1
    readonly normal: readonly [number, number, number];    // Tangent normal vector (normalized)
    readonly roughness: number;                            // 0..1
    readonly metalness: number;                            // 0..1
    readonly ao: number;                                   // 0..1
}

export interface LatentFeatureGrid {
    readonly width: number;
    readonly height: number;
    /** Channels per texel (defaults to 4 when omitted; 8 for standard8 / quality8) */
    readonly channels?: number;
    /** Row-major packed texel data (width * height * channels) */
    readonly data: Uint8Array | Float32Array;
    readonly isNormalizedU8: boolean;
}

/**
 * Validates and parses an unknown manifest payload with strict structural checks.
 */
export function parseManifest(input: unknown): NeuralMaterialManifest {
    if (!input || typeof input !== 'object') {
        throw new Error('Manifest must be a non-null object');
    }
    const m = input as Record<string, unknown>;

    if (m.format !== 'VNM1') {
        throw new Error(`Invalid manifest format: expected 'VNM1', got '${String(m.format)}'`);
    }
    if (m.version !== 1) {
        throw new Error(`Unsupported manifest version: expected 1, got '${String(m.version)}'`);
    }
    if (typeof m.assetId !== 'string' || !m.assetId.trim()) {
        throw new Error('Manifest missing or empty assetId');
    }
    if (
        !Array.isArray(m.sourceSize) ||
        m.sourceSize.length !== 2 ||
        typeof m.sourceSize[0] !== 'number' ||
        typeof m.sourceSize[1] !== 'number' ||
        !Number.isSafeInteger(m.sourceSize[0]) ||
        !Number.isSafeInteger(m.sourceSize[1]) ||
        m.sourceSize[0] <= 0 ||
        m.sourceSize[1] <= 0
    ) {
        throw new Error('Manifest sourceSize must be [width, height] array of positive integers');
    }
    const sourceW = m.sourceSize[0] as number;
    const sourceH = m.sourceSize[1] as number;

    if (m.profile !== 'small' && m.profile !== 'standard' && m.profile !== 'quality' && m.profile !== 'standard8' && m.profile !== 'quality8') {
        throw new Error(`Invalid profile: '${String(m.profile)}'`);
    }
    if (m.quantization !== 'u8') {
        throw new Error(`Unsupported quantization: '${String(m.quantization)}': only 'u8' is supported`);
    }
    if (m.weightStoragePrecision !== 'f32') {
        throw new Error(`Unsupported weightStoragePrecision: '${String(m.weightStoragePrecision)}': only 'f32' is supported`);
    }
    if (m.computePrecision !== 'f32') {
        throw new Error(`Unsupported computePrecision: '${String(m.computePrecision)}': only 'f32' is supported`);
    }
    if (m.channels !== 'base3-normal2-rough-metal-ao') {
        throw new Error(`Unsupported channels convention: '${String(m.channels)}'`);
    }
    if (m.baseColorSpace !== 'linear-srgb') {
        throw new Error(`Unsupported baseColorSpace: '${String(m.baseColorSpace)}'`);
    }
    if (m.normalConvention !== 'tangent-positive-y-positive-z') {
        throw new Error(`Unsupported normalConvention: '${String(m.normalConvention)}'`);
    }
    if (m.wrapU !== 'repeat' && m.wrapU !== 'clamp') {
        throw new Error(`Invalid wrapU: '${String(m.wrapU)}'`);
    }
    if (m.wrapV !== 'repeat' && m.wrapV !== 'clamp') {
        throw new Error(`Invalid wrapV: '${String(m.wrapV)}'`);
    }

    const validateChunkRef = (ref: unknown, name: string): ChunkRef => {
        if (!ref || typeof ref !== 'object') throw new Error(`Missing or invalid chunk reference '${name}'`);
        const c = ref as Record<string, unknown>;
        if (typeof c.url !== 'string' || !c.url.trim()) throw new Error(`Invalid url in chunk '${name}'`);
        if (typeof c.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(c.sha256)) {
            throw new Error(`Invalid sha256 in chunk '${name}': must be 64-character hexadecimal string`);
        }
        if (typeof c.byteLength !== 'number' || !Number.isSafeInteger(c.byteLength) || c.byteLength <= 0 || c.byteLength > 0xffffffff + 16) {
            throw new Error(`Invalid byteLength in chunk '${name}': must be a positive integer`);
        }
        return { url: c.url, sha256: c.sha256.toLowerCase(), byteLength: c.byteLength };
    };

    const weights = validateChunkRef(m.weights, 'weights');
    const tail = validateChunkRef(m.tail, 'tail');

    if (!Array.isArray(m.levels) || m.levels.length === 0) {
        throw new Error('Manifest levels must be a non-empty array');
    }

    const validateDim = (arr: unknown, name: string): [number, number] => {
        if (
            !Array.isArray(arr) ||
            arr.length !== 2 ||
            typeof arr[0] !== 'number' ||
            typeof arr[1] !== 'number' ||
            !Number.isSafeInteger(arr[0]) ||
            !Number.isSafeInteger(arr[1]) ||
            arr[0] <= 0 ||
            arr[1] <= 0
        ) {
            throw new Error(`Invalid ${name}: must be [width, height] array of positive integers`);
        }
        return [arr[0], arr[1]];
    };

    const levels: LatentLevel[] = m.levels.map((lvl: unknown, idx: number) => {
        if (!lvl || typeof lvl !== 'object') throw new Error(`Invalid level at index ${idx}`);
        const l = lvl as Record<string, unknown>;
        if (typeof l.sourceMip !== 'number' || !Number.isSafeInteger(l.sourceMip) || l.sourceMip < 0) {
            throw new Error(`Missing or non-integer sourceMip at level ${idx}`);
        }
        const levelSourceSize = validateDim(l.sourceSize, `levels[${idx}].sourceSize`);
        const fineSize = validateDim(l.fineSize, `levels[${idx}].fineSize`);
        const coarseSize = validateDim(l.coarseSize, `levels[${idx}].coarseSize`);
        if ([...fineSize, ...coarseSize].some((n) => n > 65535)) {
            throw new Error(`levels[${idx}]: latent dimensions must fit uint16`);
        }

        if (fineSize[0] < coarseSize[0] || fineSize[1] < coarseSize[1]) {
            throw new Error(`levels[${idx}]: fineSize must be >= coarseSize`);
        }

        const chunk = validateChunkRef(l.chunk, `levels[${idx}].chunk`);
        return {
            sourceMip: l.sourceMip,
            sourceSize: levelSourceSize,
            fineSize,
            coarseSize,
            chunk,
        };
    });

    validateManifestLevels(levels);

    if (typeof m.tailFirstMip !== 'number' || !Number.isSafeInteger(m.tailFirstMip) || m.tailFirstMip < 0) {
        throw new Error(`Invalid tailFirstMip: '${String(m.tailFirstMip)}': must be a non-negative integer`);
    }

    let mipMax: number | undefined;
    if (m.mipMax !== undefined) {
        if (typeof m.mipMax !== 'number' || !Number.isFinite(m.mipMax) || m.mipMax <= 0) {
            throw new Error(`Invalid mipMax: '${String(m.mipMax)}': must be a positive finite number`);
        }
        mipMax = m.mipMax;
    }

    return {
        format: 'VNM1',
        version: 1,
        assetId: m.assetId,
        sourceSize: [sourceW, sourceH],
        profile: m.profile,
        quantization: m.quantization,
        weightStoragePrecision: m.weightStoragePrecision,
        computePrecision: m.computePrecision,
        channels: 'base3-normal2-rough-metal-ao',
        baseColorSpace: 'linear-srgb',
        normalConvention: 'tangent-positive-y-positive-z',
        wrapU: m.wrapU,
        wrapV: m.wrapV,
        weights,
        levels,
        tailFirstMip: m.tailFirstMip,
        ...(mipMax !== undefined ? { mipMax } : {}),
        tail,
        fallbackMaterialUrl: typeof m.fallbackMaterialUrl === 'string' ? m.fallbackMaterialUrl : undefined,
    };
}
