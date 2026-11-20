/**
 * VNM1 Exporter Pipeline
 * Quantizes trained latent grids to normalized u8, serializes chunks using binary.ts,
 * calculates SHA-256 hashes via Web Crypto, and outputs a complete NeuralMaterialManifest & binary bundle.
 */

import {
  serializeWeightsChunk,
  serializeLatentsChunk,
  writeVnmHeader,
  VnmChunkKind,
} from '../format/binary';
import type {
  LatentFeatureGrid,
  NeuralMaterialManifest,
} from '../format/types';
import { getLatentChannelCount, parseManifest } from '../format/types';
import type { TrainingResult } from '../training/NeuralMaterialTrainer';
import type { PbrTextureData } from '../training/SyntheticTextureGenerator';
import { reconstructNormal } from '../training/LossFunctions';
import { validatePbrTexture } from '../training/validatePbrTexture';
import { validateLatentDimensions, validateMlpWeights } from '../format/validation';

export interface ExportOptions {
  readonly assetId?: string;
  readonly sourceWidth?: number;
  readonly sourceHeight?: number;
  readonly wrapU?: 'repeat' | 'clamp';
  readonly wrapV?: 'repeat' | 'clamp';
}

export interface ExportedChunk {
  readonly filename: string;
  readonly data: Uint8Array;
  readonly sha256: string;
  readonly byteLength: number;
}

export interface ExportedVnmPackage {
  readonly manifest: NeuralMaterialManifest;
  readonly manifestJson: string;
  readonly weightsChunk: ExportedChunk;
  readonly level0Chunk: ExportedChunk;
  readonly tailChunk: ExportedChunk;
  readonly fineGridU8: LatentFeatureGrid;
  readonly coarseGridU8: LatentFeatureGrid;
}

/**
 * Computes hexadecimal SHA-256 checksum using standard Web Crypto API
 */
export async function computeSha256Hex(data: Uint8Array): Promise<string> {
  const cryptoObj =
    typeof globalThis !== 'undefined' && globalThis.crypto
      ? globalThis.crypto
      : undefined;

  if (!cryptoObj || !cryptoObj.subtle) {
    throw new Error('Web Crypto API (crypto.subtle) is required for VNM export');
  }

  // Copy data to ensure array buffer compatibility
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);

  const hashBuffer = await cryptoObj.subtle.digest('SHA-256', copy);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Quantizes float32 latent feature values in [-1, 1] to normalized Uint8Array [0, 255]
 * Formula: u8 = clamp(round((f + 1) / 2 * 255), 0, 255)
 * Strictly verifies finiteness of every element before quantization.
 */
export function quantizeFloatToU8Grid(
  floatData: Float32Array,
  width: number,
  height: number,
  channels = 4,
): LatentFeatureGrid {
  validateLatentDimensions(width, height);
  if (!Number.isInteger(channels) || channels <= 0) {
    throw new Error(`Invalid latent channel count: ${channels}`);
  }
  const totalElements = width * height * channels;
  if (floatData.length !== totalElements) {
    throw new Error(
      `Grid dimension mismatch: expected ${totalElements} floats, got ${floatData.length}`,
    );
  }

  const u8Data = new Uint8Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    const f = floatData[i];
    if (!Number.isFinite(f)) {
      throw new Error(`Non-finite latent float value at index ${i}: ${f}`);
    }
    const normalized = (f + 1.0) * 0.5; // map [-1, 1] to [0, 1]
    const q = Math.round(normalized * 255.0);
    u8Data[i] = Math.max(0, Math.min(255, q));
  }

  return {
    width,
    height,
    channels,
    isNormalizedU8: true,
    data: u8Data,
  };
}

export function deriveTailLayout(width: number, height: number): { width: number; height: number; mip: number } {
  if (![width, height].every((n) => Number.isSafeInteger(n) && n > 0)) throw new Error('Invalid source dimensions');
  let mip = 0;
  let w = width, h = height;
  while (Math.max(w, h) > 16) {
    mip++;
    w = Math.max(1, Math.floor(width / 2 ** mip));
    h = Math.max(1, Math.floor(height / 2 ** mip));
  }
  return { width: w, height: h, mip };
}

/**
 * Creates a downscaled PBR tail chunk from the source texture using a true box downsampling filter
 * and normal vector renormalization.
 */
export function createTailChunk(
  source: PbrTextureData,
  targetWidth?: number,
  targetHeight?: number,
  tailFirstMip?: number,
): Uint8Array {
  validatePbrTexture(source);
  const layout = deriveTailLayout(source.width, source.height);
  const tailWidth = targetWidth ?? layout.width;
  const tailHeight = targetHeight ?? layout.height;
  const derivedFirstMip = tailFirstMip ?? layout.mip;
  validateLatentDimensions(tailWidth, tailHeight);
  if (!Number.isInteger(derivedFirstMip) || derivedFirstMip < 0 || derivedFirstMip > Math.floor(Math.log2(Math.max(source.width, source.height))) ||
      tailWidth !== Math.max(1, Math.floor(source.width / 2 ** derivedFirstMip)) ||
      tailHeight !== Math.max(1, Math.floor(source.height / 2 ** derivedFirstMip))) {
    throw new Error('Tail dimensions must match the selected source mip');
  }
  const tailBytes = tailWidth * tailHeight * 8; // 8 channels, 1 byte each (unorm u8)
  const payloadBytes = 16 + tailBytes; // 16 bytes subheader + payload

  const header = writeVnmHeader(VnmChunkKind.Tail, payloadBytes);
  const out = new Uint8Array(16 + payloadBytes);
  out.set(header, 0);

  const view = new DataView(out.buffer, out.byteOffset + 16, 16);
  view.setUint16(0, tailWidth, true);
  view.setUint16(2, tailHeight, true);
  view.setUint8(4, Math.max(0, Math.min(255, derivedFirstMip)));
  view.setUint8(5, 8); // 8 channels
  view.setUint8(6, 1); // 1 = u8 unorm
  // bytes 7..15 reserved = 0

  const pixelData = new Uint8Array(out.buffer, out.byteOffset + 32, tailBytes);

  // True box-filter downsampling with 3D normal vector renormalization
  for (let ty = 0; ty < tailHeight; ty++) {
    const y0 = (ty / tailHeight) * source.height;
    const y1 = ((ty + 1) / tailHeight) * source.height;

    for (let tx = 0; tx < tailWidth; tx++) {
      const x0 = (tx / tailWidth) * source.width;
      const x1 = ((tx + 1) / tailWidth) * source.width;
      const numSamples = (x1 - x0) * (y1 - y0);

      let sumR = 0, sumG = 0, sumB = 0;
      let sumNx = 0, sumNy = 0, sumNz = 0;
      let sumRough = 0, sumMetal = 0, sumAo = 0;

      for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          const area = (Math.min(sx + 1, x1) - Math.max(sx, x0)) * (Math.min(sy + 1, y1) - Math.max(sy, y0));
          const offset = (sy * source.width + sx) * 8;
          sumR += source.data[offset + 0] * area;
          sumG += source.data[offset + 1] * area;
          sumB += source.data[offset + 2] * area;

          const [nx, ny, nz] = reconstructNormal(source.data[offset + 3], source.data[offset + 4]);
          sumNx += nx * area;
          sumNy += ny * area;
          sumNz += nz * area;

          sumRough += source.data[offset + 5] * area;
          sumMetal += source.data[offset + 6] * area;
          sumAo += source.data[offset + 7] * area;
        }
      }

      const inv = 1.0 / numSamples;
      const avgR = sumR * inv;
      const avgG = sumG * inv;
      const avgB = sumB * inv;

      // Renormalize accumulated 3D normal vector
      const normLen = Math.hypot(sumNx, sumNy, sumNz) || 1.0;
      const renormNx = sumNx / normLen;
      const renormNy = sumNy / normLen;

      const avgRough = sumRough * inv;
      const avgMetal = sumMetal * inv;
      const avgAo = sumAo * inv;

      const dstOffset = (ty * tailWidth + tx) * 8;
      pixelData[dstOffset + 0] = Math.max(0, Math.min(255, Math.round(avgR * 255.0)));
      pixelData[dstOffset + 1] = Math.max(0, Math.min(255, Math.round(avgG * 255.0)));
      pixelData[dstOffset + 2] = Math.max(0, Math.min(255, Math.round(avgB * 255.0)));
      // Normal xy in [-1, 1] encoded as (val + 1) / 2 * 255
      pixelData[dstOffset + 3] = Math.max(0, Math.min(255, Math.round((renormNx + 1.0) * 0.5 * 255.0)));
      pixelData[dstOffset + 4] = Math.max(0, Math.min(255, Math.round((renormNy + 1.0) * 0.5 * 255.0)));
      pixelData[dstOffset + 5] = Math.max(0, Math.min(255, Math.round(avgRough * 255.0)));
      pixelData[dstOffset + 6] = Math.max(0, Math.min(255, Math.round(avgMetal * 255.0)));
      pixelData[dstOffset + 7] = Math.max(0, Math.min(255, Math.round(avgAo * 255.0)));
    }
  }

  return out;
}

/**
 * Exports a trained neural material to a complete VNM1 package with manifest and chunks.
 */
export async function exportNeuralMaterial(
  trainingResult: TrainingResult,
  sourceTexture: PbrTextureData,
  options: ExportOptions = {},
): Promise<ExportedVnmPackage> {
  validatePbrTexture(sourceTexture);
  validateMlpWeights(trainingResult.weights);
  const profile = trainingResult.weights.profile;
  const latentChannels = getLatentChannelCount(profile);
  const assetId = options.assetId ?? `vnm-mat-${Date.now().toString(36)}`;
  // Strict preservation of training dimensions and wrap options
  const sourceWidth = options.sourceWidth ?? trainingResult.sourceWidth ?? sourceTexture.width;
  const sourceHeight = options.sourceHeight ?? trainingResult.sourceHeight ?? sourceTexture.height;
  const wrapU = options.wrapU ?? trainingResult.wrapU ?? 'repeat';
  const wrapV = options.wrapV ?? trainingResult.wrapV ?? 'repeat';
  if (sourceWidth !== sourceTexture.width || sourceHeight !== sourceTexture.height ||
      sourceWidth !== trainingResult.sourceWidth || sourceHeight !== trainingResult.sourceHeight ||
      wrapU !== trainingResult.wrapU || wrapV !== trainingResult.wrapV) {
    throw new Error('Export metadata contradicts training or source texture');
  }

  // Strict verification of MLP weights finiteness before export
  const weightArrays = [
    trainingResult.weights.w1,
    trainingResult.weights.b1,
    trainingResult.weights.w2,
    trainingResult.weights.b2,
    trainingResult.weights.w3,
    trainingResult.weights.b3,
  ];
  for (const arr of weightArrays) {
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        throw new Error(`Non-finite weight detected during export at index ${i}: ${arr[i]}`);
      }
    }
  }

  // 1. Quantize Fine and Coarse grids to normalized u8
  const fineGridU8 = quantizeFloatToU8Grid(
    trainingResult.fineGridFloat,
    trainingResult.fineWidth,
    trainingResult.fineHeight,
    latentChannels,
  );
  const coarseGridU8 = quantizeFloatToU8Grid(
    trainingResult.coarseGridFloat,
    trainingResult.coarseWidth,
    trainingResult.coarseHeight,
    latentChannels,
  );

  // Snapshot all source-dependent binary data before asynchronous hashing yields.
  const { width: tailWidth, height: tailHeight, mip: tailFirstMip } = deriveTailLayout(sourceWidth, sourceHeight);
  const tailBin = createTailChunk(sourceTexture, tailWidth, tailHeight, tailFirstMip);

  // 2. Serialize weights chunk
  const weightsBin = serializeWeightsChunk(trainingResult.weights);
  const weightsHash = await computeSha256Hex(weightsBin);
  const weightsChunk: ExportedChunk = {
    filename: 'weights.bin',
    data: weightsBin,
    sha256: weightsHash,
    byteLength: weightsBin.byteLength,
  };

  // 3. Serialize level 0 latents chunk
  const level0Bin = serializeLatentsChunk(fineGridU8, coarseGridU8, profile);
  const level0Hash = await computeSha256Hex(level0Bin);
  const level0Chunk: ExportedChunk = {
    filename: 'level-0.bin',
    data: level0Bin,
    sha256: level0Hash,
    byteLength: level0Bin.byteLength,
  };

  // 4. Derive tail dimensions & serialize tail chunk with box filter
  const tailHash = await computeSha256Hex(tailBin);
  const tailChunk: ExportedChunk = {
    filename: 'tail.bin',
    data: tailBin,
    sha256: tailHash,
    byteLength: tailBin.byteLength,
  };

  // 5. Construct NeuralMaterialManifest (adhering strictly to Ticket A2 schema)
  const manifest: NeuralMaterialManifest = {
    format: 'VNM1',
    version: 1,
    assetId,
    sourceSize: [sourceWidth, sourceHeight],
    profile,
    quantization: 'u8',
    weightStoragePrecision: 'f32',
    computePrecision: 'f32',
    channels: 'base3-normal2-rough-metal-ao',
    baseColorSpace: 'linear-srgb',
    normalConvention: 'tangent-positive-y-positive-z',
    wrapU,
    wrapV,
    weights: {
      url: weightsChunk.filename,
      sha256: weightsChunk.sha256,
      byteLength: weightsChunk.byteLength,
    },
    levels: [
      {
        sourceMip: 0,
        sourceSize: [sourceWidth, sourceHeight],
        fineSize: [trainingResult.fineWidth, trainingResult.fineHeight],
        coarseSize: [trainingResult.coarseWidth, trainingResult.coarseHeight],
        chunk: {
          url: level0Chunk.filename,
          sha256: level0Chunk.sha256,
          byteLength: level0Chunk.byteLength,
        },
      },
    ],
    tailFirstMip,
    tail: {
      url: tailChunk.filename,
      sha256: tailChunk.sha256,
      byteLength: tailChunk.byteLength,
    },
  };

  parseManifest(manifest);
  const manifestJson = JSON.stringify(manifest, null, 2);

  return {
    manifest,
    manifestJson,
    weightsChunk,
    level0Chunk,
    tailChunk,
    fineGridU8,
    coarseGridU8,
  };
}
