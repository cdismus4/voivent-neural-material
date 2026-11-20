import { describe, expect, it } from 'vitest';
import { computeMaterialLoss } from '../training/LossFunctions';
import { createTailChunk, exportNeuralMaterial, quantizeFloatToU8Grid } from '../exporter/VnmExporter';
import { parseTailChunk } from '../format/binary';
import { generateSyntheticPbrTexture } from '../training/SyntheticTextureGenerator';
import { NeuralMaterialTrainer } from '../training/NeuralMaterialTrainer';
import { decodeNeuralMaterialRawCpu } from '../reference/CpuReferenceDecoder';

describe('training audit edge cases', () => {
    it('reproduces training from a seed and matches the production sampler', () => {
        const source = generateSyntheticPbrTexture(8, 4);
        const config = { fineWidth: 4, fineHeight: 2, coarseWidth: 2, coarseHeight: 1, seed: 42, iterations: 3, batchSize: 4 };
        const trainer = new NeuralMaterialTrainer(config);
        const result = trainer.train(source);
        const repeated = new NeuralMaterialTrainer(config).train(source);
        expect(result.lossHistory).toEqual(repeated.lossHistory);
        expect(result.weights.w1).toEqual(repeated.weights.w1);
        for (const [u, v] of [[0, 0], [1, 1], [-0.25, 1.75], [0.37, 0.61]]) {
            const expected = decodeNeuralMaterialRawCpu(
                { width: 4, height: 2, data: result.fineGridFloat, isNormalizedU8: false },
                { width: 2, height: 1, data: result.coarseGridFloat, isNormalizedU8: false },
                result.weights, { u, v },
            );
            const actual = trainer.evaluateAtUv(u, v);
            for (let c = 0; c < 8; c++) expect(Math.abs(expected[c] - actual[c])).toBeLessThan(1e-6);
        }
    });

    it('rejects empty training and non-finite source pixels', () => {
        expect(() => new NeuralMaterialTrainer({ iterations: 0 })).toThrow();
        expect(() => new NeuralMaterialTrainer({ learningRateWeights: NaN })).toThrow();
        const source = generateSyntheticPbrTexture(1, 1);
        source.data[4] = NaN;
        expect(() => new NeuralMaterialTrainer().train(source)).toThrow();
    });
    it.each([[1.2, 0.4], [-1.2, 0.4], [0.4, 1.2], [Math.sqrt(0.99995), 0]])('differentiates clamping and near-rim normals at %s,%s', (x, y) => {
        const pred = [0, 0, 0, x, y, 0, 0, 0];
        const target = [0, 0, 0, 0.3, -0.4, 0, 0, 0];
        const grad = computeMaterialLoss(pred, target).gradPred;
        const eps = 1e-8;
        for (const channel of [3, 4]) {
            const plus = [...pred], minus = [...pred];
            plus[channel] += eps; minus[channel] -= eps;
            const numeric = (computeMaterialLoss(plus, target).totalLoss - computeMaterialLoss(minus, target).totalLoss) / (2 * eps);
            expect(Math.abs(grad[channel] - numeric)).toBeLessThan(1e-4);
        }
    });

    it.each([[256, 64], [63, 29], [1, 65]])('preserves mip geometry for %s x %s sources', (width, height) => {
        const tail = parseTailChunk(createTailChunk(generateSyntheticPbrTexture(width, height)));
        expect(tail.tailWidth).toBe(Math.max(1, Math.floor(width / 2 ** tail.tailFirstMip)));
        expect(tail.tailHeight).toBe(Math.max(1, Math.floor(height / 2 ** tail.tailFirstMip)));
        expect(Math.max(tail.tailWidth, tail.tailHeight)).toBeLessThanOrEqual(16);
    });

    it('uses fractional pixel areas for odd-sized box reduction', () => {
        const source = generateSyntheticPbrTexture(5, 1);
        for (let x = 0; x < 5; x++) source.data[x * 8] = x === 2 ? 1 : 0;
        const tail = parseTailChunk(createTailChunk(source, 2, 1, 1));
        expect(tail.pixelData[0]).toBe(51);
        expect(tail.pixelData[8]).toBe(51);
    });

    it('rejects invalid source data and unrepresentable grid dimensions', () => {
        const source = generateSyntheticPbrTexture(4, 4);
        source.data[0] = NaN;
        expect(() => createTailChunk(source)).toThrow();
        expect(() => createTailChunk(generateSyntheticPbrTexture(4, 4), 8, 8, 0)).toThrow();
        expect(() => quantizeFloatToU8Grid(new Float32Array(6), 1.5, 1)).toThrow();
    });

    it('rejects export metadata that contradicts training', async () => {
        const source = generateSyntheticPbrTexture(4, 4);
        const result = new NeuralMaterialTrainer({ fineWidth: 2, fineHeight: 2, coarseWidth: 1, coarseHeight: 1, iterations: 1, batchSize: 1, wrapU: 'clamp' }).train(source);
        await expect(exportNeuralMaterial(result, source, { wrapU: 'repeat' })).rejects.toThrow();
        await expect(exportNeuralMaterial(result, source, { sourceWidth: 8 })).rejects.toThrow();
    });
});
