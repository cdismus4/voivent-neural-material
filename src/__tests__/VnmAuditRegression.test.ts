import { describe, expect, it } from 'vitest';
import { MeshStandardNodeMaterial, NearestFilter, StorageBufferAttribute } from 'three/webgpu';
import { uv } from 'three/tsl';
import type { LatentFeatureGrid, MlpWeights } from '../format/types';
import { parseManifest } from '../format/types';
import { serializeLatentsChunk, writeVnmHeader, VnmChunkKind } from '../format/binary';
import { attachNeuralPbrNodes, createLatentDataTexture, createNeuralMaterialTslNodes, createWeightsStorageBuffer } from '../gpu/TslMlpDecoderNode';

function weights(): MlpWeights {
    return { profile: 'standard', w1: new Float32Array(288), b1: new Float32Array(24),
        w2: new Float32Array(576), b2: new Float32Array(24), w3: new Float32Array(192), b3: new Float32Array(8) };
}

function grid(width = 1, height = 1): LatentFeatureGrid {
    return { width, height, data: new Uint8Array(width * height * 4), isNormalizedU8: true };
}

describe('VNM audit boundary regressions', () => {
    it('rejects short layers before storage packing can shift all subsequent offsets', () => {
        expect(() => createWeightsStorageBuffer({ ...weights(), w1: new Float32Array(284) })).toThrow();
    });

    it('rejects non-finite weights before GPU upload', () => {
        const model = weights();
        model.b3[7] = NaN;
        expect(() => createWeightsStorageBuffer(model)).toThrow();
    });

    it.each([0, 1.5, 65536])('rejects unrepresentable latent width %s', (width) => {
        expect(() => serializeLatentsChunk(grid(width), grid())).toThrow();
        expect(() => createLatentDataTexture(grid(width))).toThrow();
    });

    it.each([-1, 0.5, NaN, 2 ** 32])('rejects truncated header payload length %s', (length) => {
        expect(() => writeVnmHeader(VnmChunkKind.Latents, length)).toThrow();
    });

    it('rejects integer and non-finite raw storage buffers of the correct shape', () => {
        const texture = createLatentDataTexture(grid());
        try {
            const build = (buffer: StorageBufferAttribute) => createNeuralMaterialTslNodes(texture, texture, buffer, { uv: uv() }, { fineSize: [1, 1] });
            expect(() => build(new StorageBufferAttribute(new Uint32Array(1112), 4))).toThrow();
            const values = new Float32Array(1112);
            values[100] = Infinity;
            expect(() => build(new StorageBufferAttribute(values, 4))).toThrow();
        } finally { texture.dispose(); }
    });

    it('rejects sampler and fine-size drift from the CPU sampling contract', () => {
        const texture = createLatentDataTexture(grid());
        try {
            expect(() => createNeuralMaterialTslNodes(texture, texture, weights(), { uv: uv() }, { fineSize: [2, 1] })).toThrow();
            texture.minFilter = NearestFilter;
            expect(() => createNeuralMaterialTslNodes(texture, texture, weights(), { uv: uv() }, { fineSize: [1, 1] })).toThrow();
        } finally { texture.dispose(); }
    });

    it('invalidates an already compiled material on attach and restores exactly once', () => {
        const texture = createLatentDataTexture(grid());
        const material = new MeshStandardNodeMaterial();
        try {
            const nodes = createNeuralMaterialTslNodes(texture, texture, weights(), { uv: uv() }, { fineSize: [1, 1] });
            const before = material.version;
            const binding = attachNeuralPbrNodes(material, nodes);
            expect(material.version).toBe(before + 1);
            expect(() => attachNeuralPbrNodes(material, nodes)).toThrow();
            binding.restore();
            binding.dispose();
            expect(material.version).toBe(before + 2);
            const next = attachNeuralPbrNodes(material, nodes);
            binding.dispose();
            expect(material.aoNode).toBe(nodes.ao);
            next.dispose();
        } finally { texture.dispose(); material.dispose(); }
    });

    it('F-3 & M-5: rejects invalid manifest levels ordering/emptiness and unsupported quantization/precisions', () => {
        const base = {
            format: 'VNM1',
            version: 1,
            assetId: 'manifest-gate-test',
            sourceSize: [512, 512],
            profile: 'standard',
            quantization: 'u8',
            weightStoragePrecision: 'f32',
            computePrecision: 'f32',
            channels: 'base3-normal2-rough-metal-ao',
            baseColorSpace: 'linear-srgb',
            normalConvention: 'tangent-positive-y-positive-z',
            wrapU: 'repeat',
            wrapV: 'repeat',
            weights: { url: 'weights.bin', sha256: '0'.repeat(64), byteLength: 4464 },
            levels: [
                {
                    sourceMip: 0,
                    sourceSize: [512, 512],
                    fineSize: [32, 32],
                    coarseSize: [16, 16],
                    chunk: { url: 'level-0.bin', sha256: '1'.repeat(64), byteLength: 5152 },
                },
            ],
            tailFirstMip: 5,
            tail: { url: 'tail.bin', sha256: '2'.repeat(64), byteLength: 2080 },
        };

        expect(parseManifest(base).levels).toHaveLength(1);
        expect(() => parseManifest({ ...base, levels: [] })).toThrow(/non-empty/i);
        expect(() => parseManifest({ ...base, levels: [{ ...base.levels[0], sourceMip: 1 }] })).toThrow(/sourceMip must equal 0/i);
        expect(() => parseManifest({
            ...base,
            levels: [
                base.levels[0],
                { ...base.levels[0], sourceMip: 0 },
            ],
        })).toThrow(/ascending sourceMip/i);
        expect(() => parseManifest({
            ...base,
            levels: [
                base.levels[0],
                { ...base.levels[0], sourceMip: 2 },
                { ...base.levels[0], sourceMip: 1 },
            ],
        })).toThrow(/ascending sourceMip/i);

        expect(() => parseManifest({ ...base, quantization: 'u4' })).toThrow(/Unsupported quantization/i);
        expect(() => parseManifest({ ...base, weightStoragePrecision: 'f16' })).toThrow(/Unsupported weightStoragePrecision/i);
        expect(() => parseManifest({ ...base, computePrecision: 'f16' })).toThrow(/Unsupported computePrecision/i);
    });
});
