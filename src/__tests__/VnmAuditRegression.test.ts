import { describe, expect, it } from 'vitest';
import { MeshStandardNodeMaterial, NearestFilter, StorageBufferAttribute } from 'three/webgpu';
import { uv } from 'three/tsl';
import type { LatentFeatureGrid, MlpWeights } from '../format/types';
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
});
