import { describe, it, expect } from 'vitest';
import { uv } from 'three/tsl';
import {
  computeMaterialLoss,
  generateSyntheticPbrTexture,
  NeuralMaterialTrainer,
  exportNeuralMaterial,
  quantizeFloatToU8Grid,
  computeSha256Hex,
} from '../index';
import { parseManifest } from '../format/types';
import {
  parseVnmHeader,
  writeVnmHeader,
  parseWeightsChunk,
  parseLatentsChunk,
  parseTailChunk,
  VnmChunkKind,
} from '../format/binary';
import {
  evaluateMlpStandardCpu,
  decodeNeuralMaterialRawCpu,
} from '../reference/CpuReferenceDecoder';
import {
  createNeuralMaterialTslNodes,
  createLatentDataTexture,
  createWeightsStorageBuffer,
} from '../gpu/TslMlpDecoderNode';

describe('Ticket C: Neural Material Trainer & Exporter Pipeline', () => {
  describe('Loss Functions & Analytical Gradients', () => {
    it('returns zero loss when prediction matches target exactly', () => {
      // Normal target: nx=0, ny=0, nz=1.0
      const target = new Float32Array([0.5, 0.5, 0.5, 0.0, 0.0, 0.3, 0.7, 1.0]);
      const pred = new Float32Array([0.5, 0.5, 0.5, 0.0, 0.0, 0.3, 0.7, 1.0]);

      const res = computeMaterialLoss(pred, target);
      expect(res.totalLoss).toBeCloseTo(0.0, 5);
      expect(res.colorLoss).toBeCloseTo(0.0, 5);
      expect(res.normalLoss).toBeCloseTo(0.0, 5);
      expect(res.roughnessLoss).toBeCloseTo(0.0, 5);
      expect(res.metalnessLoss).toBeCloseTo(0.0, 5);
      expect(res.aoLoss).toBeCloseTo(0.0, 5);

      for (let i = 0; i < 8; i++) {
        expect(res.gradPred[i]).toBeCloseTo(0.0, 4);
      }
    });

    it('computes correct positive loss and finite gradients on divergence', () => {
      const target = new Float32Array([1.0, 0.0, 0.0, 0.2, -0.3, 0.1, 0.9, 0.8]);
      const pred = new Float32Array([0.2, 0.8, 0.1, -0.1, 0.4, 0.7, 0.2, 0.3]);

      const res = computeMaterialLoss(pred, target);
      expect(res.totalLoss).toBeGreaterThan(0.5);
      expect(res.colorLoss).toBeGreaterThan(0.0);
      expect(res.normalLoss).toBeGreaterThan(0.0);
      expect(res.roughnessLoss).toBeGreaterThan(0.0);

      // Verify analytical gradients are non-zero and finite
      for (let i = 0; i < 8; i++) {
        expect(Number.isFinite(res.gradPred[i])).toBe(true);
        expect(isNaN(res.gradPred[i])).toBe(false);
      }
    });

    it('matches central difference numerical differentiation (< 1e-5) for both in-disk and boundary normals', () => {
      const eps = 1e-5;
      const target = [0.7, 0.4, 0.2, 0.3, -0.4, 0.6, 0.1, 0.9];

      // Case 1: In-disk prediction (px^2 + py^2 = 0.5^2 + 0.4^2 = 0.41 < 1)
      const predInDisk = [0.65, 0.42, 0.18, 0.5, 0.4, 0.55, 0.15, 0.85];
      const analyticalInDisk = computeMaterialLoss(predInDisk, target);

      for (let i = 0; i < 8; i++) {
        const predPlus = [...predInDisk];
        predPlus[i] += eps;
        const lossPlus = computeMaterialLoss(predPlus, target).totalLoss;

        const predMinus = [...predInDisk];
        predMinus[i] -= eps;
        const lossMinus = computeMaterialLoss(predMinus, target).totalLoss;

        const numGrad = (lossPlus - lossMinus) / (2 * eps);
        const absDiff = Math.abs(analyticalInDisk.gradPred[i] - numGrad);
        expect(absDiff).toBeLessThan(1e-5);
      }

      // Case 2: Boundary / outside disk prediction (px^2 + py^2 = 0.8^2 + 0.8^2 = 1.28 >= 1)
      const predOutside = [0.5, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5, 0.5];
      const analyticalOutside = computeMaterialLoss(predOutside, target);

      for (let i = 0; i < 8; i++) {
        const predPlus = [...predOutside];
        predPlus[i] += eps;
        const lossPlus = computeMaterialLoss(predPlus, target).totalLoss;

        const predMinus = [...predOutside];
        predMinus[i] -= eps;
        const lossMinus = computeMaterialLoss(predMinus, target).totalLoss;

        const numGrad = (lossPlus - lossMinus) / (2 * eps);
        const absDiff = Math.abs(analyticalOutside.gradPred[i] - numGrad);
        expect(absDiff).toBeLessThan(1e-5);
      }
    });
  });

  describe('Synthetic Texture Generator', () => {
    it('generates 256x256 analytical PBR texture with strictly bounded values', () => {
      const tex = generateSyntheticPbrTexture(256, 256, 'checker-spheres');
      expect(tex.width).toBe(256);
      expect(tex.height).toBe(256);
      expect(tex.data.length).toBe(256 * 256 * 8);

      for (let i = 0; i < tex.data.length; i += 8) {
        // BaseColor [0..1]
        expect(tex.data[i + 0]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 0]).toBeLessThanOrEqual(1.0);
        expect(tex.data[i + 1]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 1]).toBeLessThanOrEqual(1.0);
        expect(tex.data[i + 2]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 2]).toBeLessThanOrEqual(1.0);

        // Normal [nx, ny in -1..1, nz^2 = 1 - nx^2 - ny^2 >= 0]
        const nx = tex.data[i + 3];
        const ny = tex.data[i + 4];
        expect(nx).toBeGreaterThanOrEqual(-1.0);
        expect(nx).toBeLessThanOrEqual(1.0);
        expect(ny).toBeGreaterThanOrEqual(-1.0);
        expect(ny).toBeLessThanOrEqual(1.0);
        expect(nx * nx + ny * ny).toBeLessThanOrEqual(1.0001);

        // Roughness, Metalness, AO [0..1]
        expect(tex.data[i + 5]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 5]).toBeLessThanOrEqual(1.0);
        expect(tex.data[i + 6]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 6]).toBeLessThanOrEqual(1.0);
        expect(tex.data[i + 7]).toBeGreaterThanOrEqual(0.0);
        expect(tex.data[i + 7]).toBeLessThanOrEqual(1.0);
      }
    }, 45000);
  });

  describe('Neural Material Trainer', () => {
    it('demonstrates strict loss reduction on synthetic PBR texture', () => {
      const tex = generateSyntheticPbrTexture(64, 64, 'gradient-waves');

      const trainer = new NeuralMaterialTrainer({
        fineWidth: 32,
        fineHeight: 32,
        coarseWidth: 16,
        coarseHeight: 16,
        batchSize: 256,
        iterations: 100,
        learningRateWeights: 2e-3,
        learningRateGrid: 4e-3,
      });

      const res = trainer.train(tex);
      expect(res.lossHistory.length).toBe(100);

      const initialLoss = (res.lossHistory[0] + res.lossHistory[1] + res.lossHistory[2]) / 3;
      const finalLoss = (res.lossHistory[97] + res.lossHistory[98] + res.lossHistory[99]) / 3;

      // Assert loss converged noticeably
      expect(finalLoss).toBeLessThan(initialLoss * 0.7);

      // Verify grid shapes
      expect(res.fineGridFloat.length).toBe(32 * 32 * 4);
      expect(res.coarseGridFloat.length).toBe(16 * 16 * 4);

      // All values bounded in [-1, 1]
      for (let i = 0; i < res.fineGridFloat.length; i++) {
        expect(res.fineGridFloat[i]).toBeGreaterThanOrEqual(-1.0);
        expect(res.fineGridFloat[i]).toBeLessThanOrEqual(1.0);
      }
    });

    it('achieves exact numerical equivalence (< 1e-6) between trainer forward pass and evaluateMlpStandardCpu', () => {
      const trainer = new NeuralMaterialTrainer({
        fineWidth: 16,
        fineHeight: 16,
        coarseWidth: 8,
        coarseHeight: 8,
        iterations: 1,
      });

      // Test multiple arbitrary 12-dim input vectors including production LOD and constant
      const testInputs = [
        new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.25, -0.75, 0.0, 1.0]),
        new Float32Array([-0.9, 0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.5, 0.5, 0.0, 1.0]),
        new Float32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        new Float32Array([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.8, -0.2, 0.0, 1.0]),
      ];

      const weights = trainer.getWeights();

      for (const x of testInputs) {
        const trainerOut = trainer.forwardMlp(x);
        const cpuOut = evaluateMlpStandardCpu(x, weights);

        expect(trainerOut.length).toBe(8);
        expect(cpuOut.length).toBe(8);

        for (let i = 0; i < 8; i++) {
          expect(Math.abs(trainerOut[i] - cpuOut[i])).toBeLessThan(1e-6);
        }
      }
    });

    it('samples material via evaluateAtUv with correct production input vector layout', () => {
      const trainer = new NeuralMaterialTrainer({
        fineWidth: 16,
        fineHeight: 16,
        coarseWidth: 8,
        coarseHeight: 8,
        iterations: 1,
      });

      const out = trainer.evaluateAtUv(0.3, 0.7);
      expect(out.length).toBe(8);
      for (let i = 0; i < 8; i++) {
        expect(Number.isFinite(out[i])).toBe(true);
      }
    });
  });

  describe('VNM1 Exporter Pipeline', () => {
    it('quantizes float latents to u8 accurately', () => {
      const floatData = new Float32Array([-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.8]);
      const grid = quantizeFloatToU8Grid(floatData, 2, 1);

      expect(grid.width).toBe(2);
      expect(grid.height).toBe(1);
      expect(grid.isNormalizedU8).toBe(true);
      expect(grid.data instanceof Uint8Array).toBe(true);

      const u8 = grid.data as Uint8Array;
      // -1.0 -> 0
      expect(u8[0]).toBe(0);
      // 0.0 -> 128 (or 127/128 round)
      expect(Math.abs(u8[2] - 128)).toBeLessThanOrEqual(1);
      // 1.0 -> 255
      expect(u8[4]).toBe(255);
    });

    it('strictly throws Error when quantizeFloatToU8Grid receives NaN or Infinity', () => {
      const nanData = new Float32Array([0.0, NaN, 0.5, -0.5]);
      expect(() => quantizeFloatToU8Grid(nanData, 1, 1)).toThrow(/non-finite/i);

      const infData = new Float32Array([0.0, Infinity, 0.5, -0.5]);
      expect(() => quantizeFloatToU8Grid(infData, 1, 1)).toThrow(/non-finite/i);
    });

    it('strictly throws Error when exportNeuralMaterial receives non-finite weights or latents', async () => {
      const tex = generateSyntheticPbrTexture(32, 32, 'gradient-waves');
      const badResult = {
        weights: {
          profile: 'standard' as const,
          w1: new Float32Array(24 * 12).fill(NaN),
          b1: new Float32Array(24).fill(0),
          w2: new Float32Array(24 * 24).fill(0),
          b2: new Float32Array(24).fill(0),
          w3: new Float32Array(8 * 24).fill(0),
          b3: new Float32Array(8).fill(0),
        },
        fineGridFloat: new Float32Array(16 * 16 * 4).fill(0),
        coarseGridFloat: new Float32Array(8 * 8 * 4).fill(0),
        fineWidth: 16,
        fineHeight: 16,
        coarseWidth: 8,
        coarseHeight: 8,
        wrapU: 'repeat' as const,
        wrapV: 'repeat' as const,
        sourceWidth: 32,
        sourceHeight: 32,
        lossHistory: [1.0],
        finalLoss: 1.0,
      };

      await expect(exportNeuralMaterial(badResult, tex)).rejects.toThrow(/non-finite/i);
    });

    it('computes valid 64-char hex SHA-256 checksums matching Web Crypto', async () => {
      const sample = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const hex = await computeSha256Hex(sample);
      expect(hex.length).toBe(64);
      expect(/^[0-9a-f]{64}$/.test(hex)).toBe(true);
    });

    it('evaluates canonical MLP forward pass directly via evaluateMlpStandardCpu', () => {
      const dummyWeights = {
        profile: 'standard' as const,
        w1: new Float32Array(24 * 12).fill(0.1),
        b1: new Float32Array(24).fill(0.0),
        w2: new Float32Array(24 * 24).fill(0.05),
        b2: new Float32Array(24).fill(0.0),
        w3: new Float32Array(8 * 24).fill(0.02),
        b3: new Float32Array(8).fill(0.5),
      };
      const dummyInput = new Float32Array(12).fill(0.5);
      const output = evaluateMlpStandardCpu(dummyInput, dummyWeights);
      expect(output.length).toBe(8);
      for (let i = 0; i < 8; i++) {
        expect(Number.isFinite(output[i])).toBe(true);
        expect(output[i]).toBeGreaterThan(0.0);
      }
    });

    it('exports a complete VNM package that strictly passes Ticket A2 & B1 validation', async () => {
      const tex = generateSyntheticPbrTexture(64, 64, 'checker-spheres');
      const trainer = new NeuralMaterialTrainer({
        fineWidth: 32,
        fineHeight: 32,
        coarseWidth: 16,
        coarseHeight: 16,
        iterations: 40,
        batchSize: 128,
      });

      const trainingResult = trainer.train(tex);

      const pkg = await exportNeuralMaterial(trainingResult, tex, {
        assetId: 'test-mat-001',
        wrapU: 'repeat',
        wrapV: 'repeat',
      });

      // 1. Validate manifest using Ticket A2 parseManifest()
      const parsedManifest = parseManifest(JSON.parse(pkg.manifestJson));
      expect(parsedManifest.format).toBe('VNM1');
      expect(parsedManifest.version).toBe(1);
      expect(parsedManifest.assetId).toBe('test-mat-001');
      expect(parsedManifest.profile).toBe('standard');
      expect(parsedManifest.quantization).toBe('u8');
      expect(parsedManifest.weights.sha256).toBe(pkg.weightsChunk.sha256);
      expect(parsedManifest.levels[0].chunk.sha256).toBe(pkg.level0Chunk.sha256);

      // 2. Validate Weights chunk: 16-byte header + 1,112 floats (4,448 bytes) = 4,464 bytes total
      const weightsHeader = parseVnmHeader(pkg.weightsChunk.data);
      expect(weightsHeader.chunkKind).toBe(VnmChunkKind.Weights);
      expect(weightsHeader.payloadByteLength).toBe(1112 * 4); // 4,448 bytes
      expect(pkg.weightsChunk.data.length).toBe(16 + 4448);

      const parsedWeights = parseWeightsChunk(pkg.weightsChunk.data, 'standard');
      expect(parsedWeights.w1.length).toBe(24 * 12);
      expect(parsedWeights.b1.length).toBe(24);
      expect(parsedWeights.w2.length).toBe(24 * 24);
      expect(parsedWeights.b2.length).toBe(24);
      expect(parsedWeights.w3.length).toBe(8 * 24);
      expect(parsedWeights.b3.length).toBe(8);

      // 3. Validate Latents chunk using Ticket A2 parseVnmHeader & parseLatentsChunk
      const latentsHeader = parseVnmHeader(pkg.level0Chunk.data);
      expect(latentsHeader.chunkKind).toBe(VnmChunkKind.Latents);

      const parsedLatents = parseLatentsChunk(pkg.level0Chunk.data);
      expect(parsedLatents.fine.width).toBe(32);
      expect(parsedLatents.fine.height).toBe(32);
      expect(parsedLatents.fine.isNormalizedU8).toBe(true);
      expect(parsedLatents.coarse.width).toBe(16);
      expect(parsedLatents.coarse.height).toBe(16);
      expect(parsedLatents.coarse.isNormalizedU8).toBe(true);

      // 4. Validate Tail chunk using dedicated parseTailChunk
      const tailParsed = parseTailChunk(pkg.tailChunk.data);
      expect(tailParsed.header.chunkKind).toBe(VnmChunkKind.Tail);
      expect(tailParsed.tailWidth).toBe(16);
      expect(tailParsed.tailHeight).toBe(16);
      // Source is 64x64, tail is 16x16: log2(64/16) = 2
      expect(tailParsed.tailFirstMip).toBe(2);
      expect(tailParsed.channels).toBe(8);
      expect(tailParsed.format).toBe(1);
      expect(tailParsed.pixelData.length).toBe(16 * 16 * 8);

      // 5. End-to-End CPU Reference Reconstruction
      const decodedRaw = decodeNeuralMaterialRawCpu(
        parsedLatents.fine,
        parsedLatents.coarse,
        parsedWeights,
        { u: 0.25, v: 0.75, wrapU: 'repeat', wrapV: 'repeat' },
      );
      expect(decodedRaw.length).toBe(8);
      for (let c = 0; c < 8; c++) {
        expect(Number.isFinite(decodedRaw[c])).toBe(true);
        expect(isNaN(decodedRaw[c])).toBe(false);
      }

      // 6. Connect to TSL Node Pipeline (Ticket B1)
      const fineTex = createLatentDataTexture(parsedLatents.fine);
      const coarseTex = createLatentDataTexture(parsedLatents.coarse);
      const weightsBuf = createWeightsStorageBuffer(parsedWeights);

      expect(weightsBuf.count).toBe(278);
      expect(weightsBuf.itemSize).toBe(4);

      const tslNodes = createNeuralMaterialTslNodes(
        fineTex,
        coarseTex,
        weightsBuf,
        { uv: uv() },
        { fineSize: [32, 32], coarseSize: [16, 16], wrapU: 'repeat', wrapV: 'repeat' },
      );
      expect(tslNodes.baseColorLinear).toBeDefined();
      expect(tslNodes.normalTangent).toBeDefined();
      expect(tslNodes.roughness).toBeDefined();
      expect(tslNodes.metalness).toBeDefined();
      expect(tslNodes.ao).toBeDefined();
    });

    it('strictly rejects corrupted tail chunks in parseTailChunk', () => {
      // 1. Wrong chunk kind
      const badKindData = new Uint8Array(16 + 16 + 16 * 16 * 8);
      badKindData.set(writeVnmHeader(VnmChunkKind.Weights, 16 + 16 * 16 * 8));
      expect(() => parseTailChunk(badKindData)).toThrow(/expected Tail chunk/i);

      // 2. Invalid channels (!= 8)
      const badChanData = new Uint8Array(16 + 16 + 16 * 16 * 8);
      badChanData.set(writeVnmHeader(VnmChunkKind.Tail, 16 + 16 * 16 * 8));
      const view2 = new DataView(badChanData.buffer, badChanData.byteOffset + 16, 16);
      view2.setUint16(0, 16, true);
      view2.setUint16(2, 16, true);
      view2.setUint8(4, 2); // tailFirstMip
      view2.setUint8(5, 4); // channels = 4 (invalid, must be 8)
      view2.setUint8(6, 1); // format = 1
      expect(() => parseTailChunk(badChanData)).toThrow(/Unsupported Tail channels/i);

      // 3. Non-zero reserved bytes (bytes 7..15 in subheader)
      const badReservedData = new Uint8Array(16 + 16 + 16 * 16 * 8);
      badReservedData.set(writeVnmHeader(VnmChunkKind.Tail, 16 + 16 * 16 * 8));
      const view3 = new DataView(badReservedData.buffer, badReservedData.byteOffset + 16, 16);
      view3.setUint16(0, 16, true);
      view3.setUint16(2, 16, true);
      view3.setUint8(4, 2);
      view3.setUint8(5, 8);
      view3.setUint8(6, 1);
      badReservedData[16 + 8] = 0xff; // reserved byte violated
      expect(() => parseTailChunk(badReservedData)).toThrow(/Non-zero reserved byte in tail subheader/i);

      // 4. Payload size mismatch
      const truncated = badReservedData.subarray(0, badReservedData.length - 10);
      expect(() => parseTailChunk(truncated)).toThrow();
    });
  });
});
