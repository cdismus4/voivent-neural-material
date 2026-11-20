/**
 * Neural Material Trainer (generalized across all VNM1 DecoderProfile presets)
 * Performs joint optimization of MLP weights and multi-resolution latent feature grids
 * using mini-batch Adam.
 */

import type { DecoderProfile, MlpProfileSpec, MlpWeights } from '../format/types';
import { getLatentChannelCount, MLP_PROFILES } from '../format/types';
import { computeMaterialLoss } from './LossFunctions';
import type { PbrTextureData } from './SyntheticTextureGenerator';
import { validatePbrTexture } from './validatePbrTexture';
import { validateLatentDimensions } from '../format/validation';

export interface TrainerConfig {
  readonly profile?: DecoderProfile;
  readonly fineWidth?: number;
  readonly fineHeight?: number;
  readonly coarseWidth?: number;
  readonly coarseHeight?: number;
  readonly learningRateWeights?: number;
  readonly learningRateGrid?: number;
  readonly batchSize?: number;
  readonly iterations?: number;
  readonly seed?: number;
  readonly wrapU?: 'repeat' | 'clamp';
  readonly wrapV?: 'repeat' | 'clamp';
}

export interface TrainingResult {
  readonly weights: MlpWeights;
  readonly fineGridFloat: Float32Array;
  readonly coarseGridFloat: Float32Array;
  readonly fineWidth: number;
  readonly fineHeight: number;
  readonly coarseWidth: number;
  readonly coarseHeight: number;
  readonly wrapU: 'repeat' | 'clamp';
  readonly wrapV: 'repeat' | 'clamp';
  readonly sourceWidth: number;
  readonly sourceHeight: number;
  readonly lossHistory: readonly number[];
  readonly finalLoss: number;
}

function wrapCoord(coord: number, mode: 'repeat' | 'clamp'): number {
  if (mode === 'repeat') {
    const wrapped = coord % 1;
    return wrapped < 0 ? wrapped + 1 : wrapped;
  }
  return Math.max(0, Math.min(1, coord));
}

function getTexelIndicesAndWeights(
  u: number,
  v: number,
  w: number,
  h: number,
  wrapU: 'repeat' | 'clamp',
  wrapV: 'repeat' | 'clamp',
  channels: number,
): {
  idx00: number;
  idx10: number;
  idx01: number;
  idx11: number;
  w00: number;
  w10: number;
  w01: number;
  w11: number;
  rx: number;
  ry: number;
} {
  const uWrapped = wrapCoord(u, wrapU);
  const vWrapped = wrapCoord(v, wrapV);

  const px = uWrapped * w - 0.5;
  const py = vWrapped * h - 0.5;

  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const fx = px - x0;
  const fy = py - y0;

  const wrapX = (ix: number): number => {
    return wrapU === 'repeat' ? ((ix % w) + w) % w : Math.max(0, Math.min(w - 1, ix));
  };
  const wrapY = (iy: number): number => {
    return wrapV === 'repeat' ? ((iy % h) + h) % h : Math.max(0, Math.min(h - 1, iy));
  };

  const idx00 = (wrapY(y0) * w + wrapX(x0)) * channels;
  const idx10 = (wrapY(y0) * w + wrapX(x0 + 1)) * channels;
  const idx01 = (wrapY(y0 + 1) * w + wrapX(x0)) * channels;
  const idx11 = (wrapY(y0 + 1) * w + wrapX(x0 + 1)) * channels;

  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;

  // NTC fractional offset: r = 2 * fract(p) - 1
  const rx = 2.0 * fx - 1.0;
  const ry = 2.0 * fy - 1.0;

  return { idx00, idx10, idx01, idx11, w00, w10, w01, w11, rx, ry };
}

export class NeuralMaterialTrainer {
  private readonly config: Required<TrainerConfig>;
  private readonly spec: MlpProfileSpec;
  private readonly latentChannels: number;
  private readonly weights: MlpWeights;
  private readonly fineGrid: Float32Array;
  private readonly coarseGrid: Float32Array;

  // Adam moment buffers
  private readonly mW1: Float32Array;
  private readonly vW1: Float32Array;
  private readonly mb1: Float32Array;
  private readonly vb1: Float32Array;

  private readonly mW2: Float32Array;
  private readonly vW2: Float32Array;
  private readonly mb2: Float32Array;
  private readonly vb2: Float32Array;

  private readonly mW3: Float32Array;
  private readonly vW3: Float32Array;
  private readonly mb3: Float32Array;
  private readonly vb3: Float32Array;

  private readonly mFine: Float32Array;
  private readonly vFine: Float32Array;
  private readonly mCoarse: Float32Array;
  private readonly vCoarse: Float32Array;

  private step = 0;
  private randomState = 0;

  constructor(config: TrainerConfig = {}) {
    const profile = config.profile ?? 'standard';
    if (!Object.prototype.hasOwnProperty.call(MLP_PROFILES, profile)) {
      throw new Error(`Unsupported profile: '${String(profile)}'`);
    }
    this.spec = MLP_PROFILES[profile];
    this.latentChannels = getLatentChannelCount(profile);
    this.config = {
      profile,
      fineWidth: config.fineWidth ?? 64,
      fineHeight: config.fineHeight ?? 64,
      coarseWidth: config.coarseWidth ?? 32,
      coarseHeight: config.coarseHeight ?? 32,
      learningRateWeights: config.learningRateWeights ?? 1e-3,
      learningRateGrid: config.learningRateGrid ?? 2e-3,
      batchSize: config.batchSize ?? 512,
      iterations: config.iterations ?? 300,
      seed: config.seed ?? 1337,
      wrapU: config.wrapU ?? 'repeat',
      wrapV: config.wrapV ?? 'repeat',
    };
    validateLatentDimensions(this.config.fineWidth, this.config.fineHeight);
    validateLatentDimensions(this.config.coarseWidth, this.config.coarseHeight);
    if (this.config.fineWidth < this.config.coarseWidth || this.config.fineHeight < this.config.coarseHeight ||
        ![this.config.iterations, this.config.batchSize].every((n) => Number.isSafeInteger(n) && n > 0) ||
        ![this.config.learningRateWeights, this.config.learningRateGrid].every((n) => Number.isFinite(n) && n > 0) ||
        !Number.isInteger(this.config.seed) || this.config.seed < 0 || this.config.seed > 0xffffffff ||
        ![this.config.wrapU, this.config.wrapV].every((mode) => mode === 'repeat' || mode === 'clamp')) {
      throw new Error('Invalid trainer configuration');
    }
    this.randomState = this.config.seed;

    const { inputDim, hidden1, hidden2, outputDim } = this.spec;
    this.weights = {
      profile,
      w1: new Float32Array(hidden1 * inputDim),
      b1: new Float32Array(hidden1),
      w2: new Float32Array(hidden2 * hidden1),
      b2: new Float32Array(hidden2),
      w3: new Float32Array(outputDim * hidden2),
      b3: new Float32Array(outputDim),
    };

    // Initialize MLP weights with He/Xavier uniform
    this.initUniform(this.weights.w1, Math.sqrt(2.0 / inputDim));
    this.initUniform(this.weights.w2, Math.sqrt(2.0 / hidden1));
    this.initUniform(this.weights.w3, Math.sqrt(2.0 / hidden2));

    // Grids
    const fineSize = this.config.fineWidth * this.config.fineHeight * this.latentChannels;
    const coarseSize = this.config.coarseWidth * this.config.coarseHeight * this.latentChannels;
    this.fineGrid = new Float32Array(fineSize);
    this.coarseGrid = new Float32Array(coarseSize);

    // Initialize grids with small random floats in [-0.1, 0.1]
    this.initUniform(this.fineGrid, 0.1);
    this.initUniform(this.coarseGrid, 0.1);

    // Adam buffers
    this.mW1 = new Float32Array(hidden1 * inputDim);
    this.vW1 = new Float32Array(hidden1 * inputDim);
    this.mb1 = new Float32Array(hidden1);
    this.vb1 = new Float32Array(hidden1);

    this.mW2 = new Float32Array(hidden2 * hidden1);
    this.vW2 = new Float32Array(hidden2 * hidden1);
    this.mb2 = new Float32Array(hidden2);
    this.vb2 = new Float32Array(hidden2);

    this.mW3 = new Float32Array(outputDim * hidden2);
    this.vW3 = new Float32Array(outputDim * hidden2);
    this.mb3 = new Float32Array(outputDim);
    this.vb3 = new Float32Array(outputDim);

    this.mFine = new Float32Array(fineSize);
    this.vFine = new Float32Array(fineSize);
    this.mCoarse = new Float32Array(coarseSize);
    this.vCoarse = new Float32Array(coarseSize);
  }

  private initUniform(arr: Float32Array, scale: number): void {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = (this.nextRandom() * 2.0 - 1.0) * scale;
    }
  }

  private nextRandom(): number {
    this.randomState = (Math.imul(this.randomState, 1664525) + 1013904223) >>> 0;
    return this.randomState / 4294967296;
  }

  public train(
    texture: PbrTextureData,
    onProgress?: (iter: number, loss: number) => void,
  ): TrainingResult {
    validatePbrTexture(texture);
    const { width, height, data } = texture;
    const { batchSize, iterations, wrapU, wrapV } = this.config;
    const lossHistory: number[] = [];

    const { inputDim, hidden1, hidden2, outputDim } = this.spec;
    const ch = this.latentChannels;
    const latentInputCount = ch * 2;

    // Preallocate gradient accumulators for one batch
    const gradW1 = new Float32Array(hidden1 * inputDim);
    const gradb1 = new Float32Array(hidden1);
    const gradW2 = new Float32Array(hidden2 * hidden1);
    const gradb2 = new Float32Array(hidden2);
    const gradW3 = new Float32Array(outputDim * hidden2);
    const gradb3 = new Float32Array(outputDim);
    const gradFine = new Float32Array(this.fineGrid.length);
    const gradCoarse = new Float32Array(this.coarseGrid.length);

    // Scratch buffers for forward/backward pass per sample
    const x = new Float32Array(inputDim);
    const h1 = new Float32Array(hidden1);
    const z1 = new Float32Array(hidden1); // pre-relu
    const h2 = new Float32Array(hidden2);
    const z2 = new Float32Array(hidden2); // pre-relu
    const yPred = new Float32Array(outputDim);

    const gradZ2 = new Float32Array(hidden2);
    const gradH1 = new Float32Array(hidden1);
    const gradZ1 = new Float32Array(hidden1);
    const gradX = new Float32Array(inputDim);

    for (let iter = 0; iter < iterations; iter++) {
      this.step++;
      gradW1.fill(0);
      gradb1.fill(0);
      gradW2.fill(0);
      gradb2.fill(0);
      gradW3.fill(0);
      gradb3.fill(0);
      gradFine.fill(0);
      gradCoarse.fill(0);

      let batchLoss = 0;

      for (let b = 0; b < batchSize; b++) {
        // Sample random texel
        const px = Math.floor(this.nextRandom() * width);
        const py = Math.floor(this.nextRandom() * height);
        const u = (px + 0.5) / width;
        const v = (py + 0.5) / height;

        // Target 8 values
        const targetOffset = (py * width + px) * 8;
        const target = data.subarray(targetOffset, targetOffset + 8);

        // Bilinear sample fine
        const fineInfo = getTexelIndicesAndWeights(
          u,
          v,
          this.config.fineWidth,
          this.config.fineHeight,
          wrapU,
          wrapV,
          ch,
        );
        for (let c = 0; c < ch; c++) {
          x[c] =
            this.fineGrid[fineInfo.idx00 + c] * fineInfo.w00 +
            this.fineGrid[fineInfo.idx10 + c] * fineInfo.w10 +
            this.fineGrid[fineInfo.idx01 + c] * fineInfo.w01 +
            this.fineGrid[fineInfo.idx11 + c] * fineInfo.w11;
        }

        // Bilinear sample coarse
        const coarseInfo = getTexelIndicesAndWeights(
          u,
          v,
          this.config.coarseWidth,
          this.config.coarseHeight,
          wrapU,
          wrapV,
          ch,
        );
        for (let c = 0; c < ch; c++) {
          x[ch + c] =
            this.coarseGrid[coarseInfo.idx00 + c] * coarseInfo.w00 +
            this.coarseGrid[coarseInfo.idx10 + c] * coarseInfo.w10 +
            this.coarseGrid[coarseInfo.idx01 + c] * coarseInfo.w01 +
            this.coarseGrid[coarseInfo.idx11 + c] * coarseInfo.w11;
        }

        // Fractional positions & production context:
        // x[2*ch..2*ch+1]: Fine grid texel local offset r = 2 * fract(p) - 1
        // x[2*ch+2]: normalizedMip (0.0 for level 0 / 1-mip trial)
        // x[2*ch+3]: constant bias 1.0 (production shader standard)
        x[latentInputCount + 0] = fineInfo.rx;
        x[latentInputCount + 1] = fineInfo.ry;
        x[latentInputCount + 2] = 0.0;
        x[latentInputCount + 3] = 1.0;

        this.forwardInto(x, h1, h2, yPred, z1, z2);

        // Compute loss and analytical gradient
        const loss = computeMaterialLoss(yPred, target);
        batchLoss += loss.totalLoss;
        const gradY = loss.gradPred;

        // Backward: Layer 3
        for (let i = 0; i < outputDim; i++) {
          const gy = gradY[i];
          gradb3[i] += gy;
          const row = i * hidden2;
          for (let j = 0; j < hidden2; j++) {
            gradW3[row + j] += gy * h2[j];
          }
        }

        // Backward: Layer 2
        gradZ2.fill(0);
        for (let j = 0; j < hidden2; j++) {
          if (h2[j] > 0) {
            let sum = 0;
            for (let i = 0; i < outputDim; i++) {
              sum += this.weights.w3[i * hidden2 + j] * gradY[i];
            }
            gradZ2[j] = sum;
          }
        }

        for (let i = 0; i < hidden2; i++) {
          const gz = gradZ2[i];
          gradb2[i] += gz;
          const row = i * hidden1;
          for (let j = 0; j < hidden1; j++) {
            gradW2[row + j] += gz * h1[j];
          }
        }

        // Backward: Layer 1
        gradH1.fill(0);
        for (let j = 0; j < hidden1; j++) {
          let sum = 0;
          for (let i = 0; i < hidden2; i++) {
            sum += this.weights.w2[i * hidden1 + j] * gradZ2[i];
          }
          gradH1[j] = sum;
        }

        gradZ1.fill(0);
        for (let j = 0; j < hidden1; j++) {
          if (h1[j] > 0) {
            gradZ1[j] = gradH1[j];
          }
        }

        for (let i = 0; i < hidden1; i++) {
          const gz = gradZ1[i];
          gradb1[i] += gz;
          const row = i * inputDim;
          for (let j = 0; j < inputDim; j++) {
            gradW1[row + j] += gz * x[j];
          }
        }

        // Backward to inputs x (first 2*ch channels = latents)
        gradX.fill(0);
        for (let j = 0; j < latentInputCount; j++) {
          let sum = 0;
          for (let i = 0; i < hidden1; i++) {
            sum += this.weights.w1[i * inputDim + j] * gradZ1[i];
          }
          gradX[j] = sum;
        }

        // Distribute gradX[0..ch-1] to fineGrid texels
        for (let c = 0; c < ch; c++) {
          const g = gradX[c];
          gradFine[fineInfo.idx00 + c] += g * fineInfo.w00;
          gradFine[fineInfo.idx10 + c] += g * fineInfo.w10;
          gradFine[fineInfo.idx01 + c] += g * fineInfo.w01;
          gradFine[fineInfo.idx11 + c] += g * fineInfo.w11;
        }

        // Distribute gradX[ch..2*ch-1] to coarseGrid texels
        for (let c = 0; c < ch; c++) {
          const g = gradX[ch + c];
          gradCoarse[coarseInfo.idx00 + c] += g * coarseInfo.w00;
          gradCoarse[coarseInfo.idx10 + c] += g * coarseInfo.w10;
          gradCoarse[coarseInfo.idx01 + c] += g * coarseInfo.w01;
          gradCoarse[coarseInfo.idx11 + c] += g * coarseInfo.w11;
        }
      }

      batchLoss /= batchSize;
      lossHistory.push(batchLoss);

      // Average gradients by batchSize
      const invBatch = 1.0 / batchSize;
      this.scaleInPlace(gradW1, invBatch);
      this.scaleInPlace(gradb1, invBatch);
      this.scaleInPlace(gradW2, invBatch);
      this.scaleInPlace(gradb2, invBatch);
      this.scaleInPlace(gradW3, invBatch);
      this.scaleInPlace(gradb3, invBatch);
      this.scaleInPlace(gradFine, invBatch);
      this.scaleInPlace(gradCoarse, invBatch);

      // Adam Step
      const lrW = this.config.learningRateWeights;
      const lrG = this.config.learningRateGrid;
      this.adamUpdate(this.weights.w1, gradW1, this.mW1, this.vW1, lrW);
      this.adamUpdate(this.weights.b1, gradb1, this.mb1, this.vb1, lrW);
      this.adamUpdate(this.weights.w2, gradW2, this.mW2, this.vW2, lrW);
      this.adamUpdate(this.weights.b2, gradb2, this.mb2, this.vb2, lrW);
      this.adamUpdate(this.weights.w3, gradW3, this.mW3, this.vW3, lrW);
      this.adamUpdate(this.weights.b3, gradb3, this.mb3, this.vb3, lrW);

      this.adamUpdate(this.fineGrid, gradFine, this.mFine, this.vFine, lrG);
      this.adamUpdate(this.coarseGrid, gradCoarse, this.mCoarse, this.vCoarse, lrG);

      // Clamp grid values to [-1, 1]
      this.clampGrid(this.fineGrid);
      this.clampGrid(this.coarseGrid);

      if (onProgress && (iter % 20 === 0 || iter === iterations - 1)) {
        onProgress(iter, batchLoss);
      }
    }

    return {
      weights: this.weights,
      fineGridFloat: this.fineGrid,
      coarseGridFloat: this.coarseGrid,
      fineWidth: this.config.fineWidth,
      fineHeight: this.config.fineHeight,
      coarseWidth: this.config.coarseWidth,
      coarseHeight: this.config.coarseHeight,
      wrapU: this.config.wrapU,
      wrapV: this.config.wrapV,
      sourceWidth: width,
      sourceHeight: height,
      lossHistory,
      finalLoss: lossHistory[lossHistory.length - 1],
    };
  }

  /**
   * Returns a reference to the current MLP weights.
   */
  public getWeights(): MlpWeights {
    return this.weights;
  }

  /**
   * Evaluates the MLP forward pass directly on a given inputDim-dim input vector.
   * Matches evaluateMlpStandardCpu exactly.
   */
  public forwardMlp(x: ArrayLike<number>): Float32Array {
    const { inputDim, hidden1, hidden2, outputDim } = this.spec;
    if (x.length !== inputDim) {
      throw new Error(`Expected ${inputDim} inputs, got ${x.length}`);
    }
    const h1 = new Float32Array(hidden1);
    const h2 = new Float32Array(hidden2);
    const yPred = new Float32Array(outputDim);
    this.forwardInto(x, h1, h2, yPred);
    return yPred;
  }

  private forwardInto(x: ArrayLike<number>, h1: Float32Array, h2: Float32Array, yPred: Float32Array,
    z1?: Float32Array, z2?: Float32Array): void {
    const { inputDim, hidden1, hidden2, outputDim } = this.spec;
    for (let i = 0; i < hidden1; i++) {
      let sum = this.weights.b1[i];
      const row = i * inputDim;
      for (let j = 0; j < inputDim; j++) {
        sum += this.weights.w1[row + j] * x[j];
      }
      if (z1) z1[i] = sum;
      h1[i] = Math.max(0, sum);
    }
    for (let i = 0; i < hidden2; i++) {
      let sum = this.weights.b2[i];
      const row = i * hidden1;
      for (let j = 0; j < hidden1; j++) {
        sum += this.weights.w2[row + j] * h1[j];
      }
      if (z2) z2[i] = sum;
      h2[i] = Math.max(0, sum);
    }
    for (let i = 0; i < outputDim; i++) {
      let sum = this.weights.b3[i];
      const row = i * hidden2;
      for (let j = 0; j < hidden2; j++) {
        sum += this.weights.w3[row + j] * h2[j];
      }
      yPred[i] = sum;
    }
  }

  /**
   * Evaluates neural material reconstruction at (u, v) using unquantized float grids and weights.
   * Strictly aligns with production input vector layout (Fine -> Coarse -> rx,ry, 0.0, 1.0).
   */
  public evaluateAtUv(u: number, v: number): Float32Array {
    const ch = this.latentChannels;
    const fineInfo = getTexelIndicesAndWeights(
      u,
      v,
      this.config.fineWidth,
      this.config.fineHeight,
      this.config.wrapU,
      this.config.wrapV,
      ch,
    );
    const coarseInfo = getTexelIndicesAndWeights(
      u,
      v,
      this.config.coarseWidth,
      this.config.coarseHeight,
      this.config.wrapU,
      this.config.wrapV,
      ch,
    );

    const x = new Float32Array(this.spec.inputDim);
    for (let c = 0; c < ch; c++) {
      x[c] =
        this.fineGrid[fineInfo.idx00 + c] * fineInfo.w00 +
        this.fineGrid[fineInfo.idx10 + c] * fineInfo.w10 +
        this.fineGrid[fineInfo.idx01 + c] * fineInfo.w01 +
        this.fineGrid[fineInfo.idx11 + c] * fineInfo.w11;
    }
    for (let c = 0; c < ch; c++) {
      x[ch + c] =
        this.coarseGrid[coarseInfo.idx00 + c] * coarseInfo.w00 +
        this.coarseGrid[coarseInfo.idx10 + c] * coarseInfo.w10 +
        this.coarseGrid[coarseInfo.idx01 + c] * coarseInfo.w01 +
        this.coarseGrid[coarseInfo.idx11 + c] * coarseInfo.w11;
    }
    x[ch * 2 + 0] = fineInfo.rx;
    x[ch * 2 + 1] = fineInfo.ry;
    x[ch * 2 + 2] = 0.0;
    x[ch * 2 + 3] = 1.0;

    return this.forwardMlp(x);
  }

  private scaleInPlace(arr: Float32Array, scale: number): void {
    for (let i = 0; i < arr.length; i++) {
      arr[i] *= scale;
    }
  }

  private clampGrid(grid: Float32Array): void {
    for (let i = 0; i < grid.length; i++) {
      grid[i] = Math.max(-1.0, Math.min(1.0, grid[i]));
    }
  }

  private adamUpdate(
    theta: Float32Array,
    grad: Float32Array,
    m: Float32Array,
    v: Float32Array,
    lr: number,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
  ): void {
    const t = this.step;
    const correction1 = 1.0 - Math.pow(beta1, t);
    const correction2 = 1.0 - Math.pow(beta2, t);

    for (let i = 0; i < theta.length; i++) {
      const g = grad[i];
      m[i] = beta1 * m[i] + (1.0 - beta1) * g;
      v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

      const mHat = m[i] / correction1;
      const vHat = v[i] / correction2;

      theta[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }
}
