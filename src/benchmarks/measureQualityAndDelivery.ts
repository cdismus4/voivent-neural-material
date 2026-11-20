/**
 * VNM1 Quality & Delivery Measurement Benchmark
 * 
 * Re-trains and exports representative PBR textures using the audited pipeline,
 * then rigorously measures:
 * 1. Delivery volume (byte count) vs raw textures (compression ratio)
 * 2. Reconstruction image quality (RMSE, MAE, PSNR in dB) across all channels
 */

import { generateSyntheticPbrTexture } from '../training/SyntheticTextureGenerator';
import { NeuralMaterialTrainer } from '../training/NeuralMaterialTrainer';
import { exportNeuralMaterial } from '../exporter/VnmExporter';
import { parseLatentsChunk, parseWeightsChunk } from '../format/binary';
import { decodeNeuralMaterialRawCpu } from '../reference/CpuReferenceDecoder';

interface BenchmarkResult {
  pattern: 'checker-spheres' | 'gradient-waves';
  width: number;
  height: number;
  rawUnorm8Bytes: number;
  rawFloat32Bytes: number;
  vnmWeightsBytes: number;
  vnmLevel0Bytes: number;
  vnmTailBytes: number;
  vnmManifestBytes: number;
  vnmTotalBytes: number;
  compressionRatioVsUnorm8: number;
  savingsVsUnorm8Percent: number;
  trainingIterations: number;
  trainingTimeMs: number;
  rmseColor: number;
  psnrColor: number;
  rmseNormal: number;
  psnrNormal: number;
  rmseRoughness: number;
  rmseMetalness: number;
  rmseAo: number;
  rmseOverall: number;
  psnrOverall: number;
  maeOverall: number;
  maxAbsErrorOverall: number;
}

export async function runBenchmark(
  pattern: 'checker-spheres' | 'gradient-waves',
  iterations = 300,
): Promise<BenchmarkResult> {
  const width = 256;
  const height = 256;
  const source = generateSyntheticPbrTexture(width, height, pattern);

  const startTime = Date.now();
  const trainer = new NeuralMaterialTrainer({
    fineWidth: 64,
    fineHeight: 64,
    coarseWidth: 32,
    coarseHeight: 32,
    batchSize: 512,
    iterations,
    learningRateWeights: 2e-3,
    learningRateGrid: 4e-3,
    wrapU: 'repeat',
    wrapV: 'repeat',
  });

  const trainingResult = trainer.train(source);
  const trainingTimeMs = Date.now() - startTime;

  const pkg = await exportNeuralMaterial(trainingResult, source, {
    assetId: `bench-${pattern}`,
    wrapU: 'repeat',
    wrapV: 'repeat',
  });

  // 1. Delivery Size Metrics
  const rawUnorm8Bytes = width * height * 8;
  const rawFloat32Bytes = width * height * 8 * 4;
  const vnmWeightsBytes = pkg.weightsChunk.byteLength;
  const vnmLevel0Bytes = pkg.level0Chunk.byteLength;
  const vnmTailBytes = pkg.tailChunk.byteLength;
  const vnmManifestBytes = new TextEncoder().encode(pkg.manifestJson).byteLength;
  const vnmTotalBytes = vnmWeightsBytes + vnmLevel0Bytes + vnmTailBytes + vnmManifestBytes;

  const compressionRatioVsUnorm8 = rawUnorm8Bytes / vnmTotalBytes;
  const savingsVsUnorm8Percent = (1.0 - vnmTotalBytes / rawUnorm8Bytes) * 100;

  // 2. Decode full 256x256 grid via canonical CPU decoder using quantized package
  const parsedWeights = parseWeightsChunk(pkg.weightsChunk.data, 'standard');
  const parsedLatents = parseLatentsChunk(pkg.level0Chunk.data);

  let sumSqColor = 0;
  let sumSqNormal = 0;
  let sumSqRough = 0;
  let sumSqMetal = 0;
  let sumSqAo = 0;
  let sumSqOverall = 0;
  let sumAbsOverall = 0;
  let maxAbsOverall = 0;

  const totalPixels = width * height;

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const u = (px + 0.5) / width;
      const v = (py + 0.5) / height;

      const decoded = decodeNeuralMaterialRawCpu(
        parsedLatents.fine,
        parsedLatents.coarse,
        parsedWeights,
        { u, v, wrapU: 'repeat', wrapV: 'repeat' },
      );

      const offset = (py * width + px) * 8;

      // Color (channels 0..2)
      for (let c = 0; c < 3; c++) {
        const diff = decoded[c] - source.data[offset + c];
        const sq = diff * diff;
        sumSqColor += sq;
        sumSqOverall += sq;
        const abs = Math.abs(diff);
        sumAbsOverall += abs;
        if (abs > maxAbsOverall) maxAbsOverall = abs;
      }

      // Normal (channels 3..4)
      for (let c = 3; c < 5; c++) {
        const diff = decoded[c] - source.data[offset + c];
        const sq = diff * diff;
        sumSqNormal += sq;
        sumSqOverall += sq;
        const abs = Math.abs(diff);
        sumAbsOverall += abs;
        if (abs > maxAbsOverall) maxAbsOverall = abs;
      }

      // Roughness (channel 5)
      {
        const diff = decoded[5] - source.data[offset + 5];
        const sq = diff * diff;
        sumSqRough += sq;
        sumSqOverall += sq;
        const abs = Math.abs(diff);
        sumAbsOverall += abs;
        if (abs > maxAbsOverall) maxAbsOverall = abs;
      }

      // Metalness (channel 6)
      {
        const diff = decoded[6] - source.data[offset + 6];
        const sq = diff * diff;
        sumSqMetal += sq;
        sumSqOverall += sq;
        const abs = Math.abs(diff);
        sumAbsOverall += abs;
        if (abs > maxAbsOverall) maxAbsOverall = abs;
      }

      // AO (channel 7)
      {
        const diff = decoded[7] - source.data[offset + 7];
        const sq = diff * diff;
        sumSqAo += sq;
        sumSqOverall += sq;
        const abs = Math.abs(diff);
        sumAbsOverall += abs;
        if (abs > maxAbsOverall) maxAbsOverall = abs;
      }
    }
  }

  const rmseColor = Math.sqrt(sumSqColor / (totalPixels * 3));
  const psnrColor = -20 * Math.log10(Math.max(1e-6, rmseColor));

  const rmseNormal = Math.sqrt(sumSqNormal / (totalPixels * 2));
  // Normal values range in [-1, 1], dynamic range = 2.0
  const psnrNormal = 20 * Math.log10(2.0 / Math.max(1e-6, rmseNormal));

  const rmseRoughness = Math.sqrt(sumSqRough / totalPixels);
  const rmseMetalness = Math.sqrt(sumSqMetal / totalPixels);
  const rmseAo = Math.sqrt(sumSqAo / totalPixels);

  const rmseOverall = Math.sqrt(sumSqOverall / (totalPixels * 8));
  const psnrOverall = -20 * Math.log10(Math.max(1e-6, rmseOverall));
  const maeOverall = sumAbsOverall / (totalPixels * 8);

  return {
    pattern,
    width,
    height,
    rawUnorm8Bytes,
    rawFloat32Bytes,
    vnmWeightsBytes,
    vnmLevel0Bytes,
    vnmTailBytes,
    vnmManifestBytes,
    vnmTotalBytes,
    compressionRatioVsUnorm8,
    savingsVsUnorm8Percent,
    trainingIterations: iterations,
    trainingTimeMs,
    rmseColor,
    psnrColor,
    rmseNormal,
    psnrNormal,
    rmseRoughness,
    rmseMetalness,
    rmseAo,
    rmseOverall,
    psnrOverall,
    maeOverall,
    maxAbsErrorOverall: maxAbsOverall,
  };
}

async function main() {
  console.log('Running VNM1 Quality & Delivery Measurement Benchmark (300 iterations per pattern)...');

  const res1 = await runBenchmark('gradient-waves', 300);
  console.log('Gradient Waves complete:', {
    time: `${res1.trainingTimeMs}ms`,
    deliveryKB: `${(res1.vnmTotalBytes / 1024).toFixed(1)} KB`,
    ratio: `${res1.compressionRatioVsUnorm8.toFixed(1)}x`,
    overallRmse: res1.rmseOverall.toFixed(4),
    overallPsnr: `${res1.psnrOverall.toFixed(2)} dB`,
  });

  const res2 = await runBenchmark('checker-spheres', 300);
  console.log('Checker Spheres complete:', {
    time: `${res2.trainingTimeMs}ms`,
    deliveryKB: `${(res2.vnmTotalBytes / 1024).toFixed(1)} KB`,
    ratio: `${res2.compressionRatioVsUnorm8.toFixed(1)}x`,
    overallRmse: res2.rmseOverall.toFixed(4),
    overallPsnr: `${res2.psnrOverall.toFixed(2)} dB`,
  });

  console.log('\n================ BENCHMARK REPORT SUMMARY ================');
  console.log(JSON.stringify({ 'gradient-waves': res1, 'checker-spheres': res2 }, null, 2));
}

const entryArg = (globalThis as { process?: { argv?: readonly string[] } }).process?.argv?.[1] ?? '';
if (entryArg.length > 0 && import.meta.url.endsWith(entryArg)) {
  main().catch(console.error);
}
