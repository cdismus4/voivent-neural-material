import type { DecodedPbrChannels, LatentFeatureGrid, MlpWeights } from '../format/types';
import { getLatentChannelCount, MLP_PROFILES } from '../format/types';

export interface SampleContext {
    readonly u: number;
    readonly v: number;
    readonly normalizedMip?: number;
    readonly wrapU?: 'repeat' | 'clamp';
    readonly wrapV?: 'repeat' | 'clamp';
}

function wrapCoord(coord: number, mode: 'repeat' | 'clamp'): number {
    if (mode === 'repeat') {
        const wrapped = coord % 1;
        return wrapped < 0 ? wrapped + 1 : wrapped;
    }
    return Math.max(0, Math.min(1, coord));
}

function sampleGridBilinear(
    grid: LatentFeatureGrid,
    u: number,
    v: number,
    wrapU: 'repeat' | 'clamp',
    wrapV: 'repeat' | 'clamp',
    channels = grid.channels ?? 4,
): number[] {
    const w = grid.width;
    const h = grid.height;
    if (grid.channels !== undefined && grid.channels !== channels) {
        throw new Error(`Latent grid channel count mismatch: expected ${channels}, got ${grid.channels}`);
    }
    if (grid.data.length !== w * h * channels) {
        throw new Error(`Latent grid data length mismatch: expected ${w * h * channels}, got ${grid.data.length}`);
    }

    const uWrapped = wrapCoord(u, wrapU);
    const vWrapped = wrapCoord(v, wrapV);

    // Texel center convention: p = uv * size - 0.5
    const px = uWrapped * w - 0.5;
    const py = vWrapped * h - 0.5;

    const x0 = Math.floor(px);
    const y0 = Math.floor(py);
    const fx = px - x0;
    const fy = py - y0;

    const getTexel = (x: number, y: number): number[] => {
        let ix = x;
        let iy = y;
        if (wrapU === 'repeat') {
            ix = ((ix % w) + w) % w;
        } else {
            ix = Math.max(0, Math.min(w - 1, ix));
        }
        if (wrapV === 'repeat') {
            iy = ((iy % h) + h) % h;
        } else {
            iy = Math.max(0, Math.min(h - 1, iy));
        }

        const idx = (iy * w + ix) * channels;
        const vals = new Array<number>(channels);
        if (grid.isNormalizedU8) {
            const raw = grid.data as Uint8Array;
            // Dequantize: f = 2 * (q / 255) - 1
            for (let c = 0; c < channels; c++) {
                vals[c] = (raw[idx + c]! / 255) * 2 - 1;
            }
        } else {
            const raw = grid.data as Float32Array;
            for (let c = 0; c < channels; c++) {
                vals[c] = raw[idx + c]!;
            }
        }
        return vals;
    };

    const c00 = getTexel(x0, y0);
    const c10 = getTexel(x0 + 1, y0);
    const c01 = getTexel(x0, y0 + 1);
    const c11 = getTexel(x0 + 1, y0 + 1);

    const out = new Array<number>(channels);
    for (let c = 0; c < channels; c++) {
        const top = c00[c]! * (1 - fx) + c10[c]! * fx;
        const bottom = c01[c]! * (1 - fx) + c11[c]! * fx;
        out[c] = top * (1 - fy) + bottom * fy;
    }
    return out;
}

/**
 * Evaluates the pure VNM1 MLP forward pass across all registered profiles
 * (inputDim -> hidden1 -> hidden2 -> 8 raw linear outputs).
 * Accumulates dot products in float64 and stores layer activations in Float32Array.
 */
export function evaluateMlpStandardCpu(
    x: ArrayLike<number>,
    weights: MlpWeights,
): Float32Array {
    const spec = MLP_PROFILES[weights.profile];
    if (!spec) {
        throw new Error(`Unsupported profile: '${String(weights.profile)}'`);
    }
    if (x.length !== spec.inputDim) {
        throw new Error(`Expected ${spec.inputDim} inputs for profile '${weights.profile}', got ${x.length}`);
    }
    const inputDim = spec.inputDim;
    const hidden1 = weights.b1.length;
    const h1 = new Float32Array(hidden1);
    for (let i = 0; i < hidden1; i++) {
        let sum = weights.b1[i]!;
        const rowOffset = i * inputDim;
        for (let j = 0; j < inputDim; j++) {
            sum += weights.w1[rowOffset + j]! * x[j]!;
        }
        h1[i] = Math.max(0, sum); // ReLU
    }

    const hidden2 = weights.b2.length;
    const h2 = new Float32Array(hidden2);
    for (let i = 0; i < hidden2; i++) {
        let sum = weights.b2[i]!;
        const rowOffset = i * hidden1;
        for (let j = 0; j < hidden1; j++) {
            sum += weights.w2[rowOffset + j]! * h1[j]!;
        }
        h2[i] = Math.max(0, sum); // ReLU
    }

    const outputDim = weights.b3.length;
    const y = new Float32Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
        let sum = weights.b3[i]!;
        const rowOffset = i * hidden2;
        for (let j = 0; j < hidden2; j++) {
            sum += weights.w3[rowOffset + j]! * h2[j]!;
        }
        y[i] = sum;
    }
    return y;
}

/**
 * Computes the raw, un-clamped 8-dimensional output from the canonical CPU reference decoder.
 * Generalized across all profiles (4+4ch and 8+8ch) using profile-derived channel counts.
 */
export function decodeNeuralMaterialRawCpu(
    fineGrid: LatentFeatureGrid,
    coarseGrid: LatentFeatureGrid,
    weights: MlpWeights,
    ctx: SampleContext,
): Float32Array {
    const spec = MLP_PROFILES[weights.profile];
    if (!spec) {
        throw new Error(`Unsupported profile: '${String(weights.profile)}'`);
    }
    const ch = getLatentChannelCount(weights.profile);
    const wrapU = ctx.wrapU ?? 'repeat';
    const wrapV = ctx.wrapV ?? 'repeat';

    // 1. Bilinear sample Fine grid (ch channels)
    const fine = sampleGridBilinear(fineGrid, ctx.u, ctx.v, wrapU, wrapV, ch);

    // 2. Bilinear sample Coarse grid (ch channels)
    const coarse = sampleGridBilinear(coarseGrid, ctx.u, ctx.v, wrapU, wrapV, ch);

    // 3. Local coordinate inside Fine grid texel:
    // p = uv * size(Fine) - 0.5; r = 2 * fract(p) - 1
    const uWrapped = wrapCoord(ctx.u, wrapU);
    const vWrapped = wrapCoord(ctx.v, wrapV);
    const px = uWrapped * fineGrid.width - 0.5;
    const py = vWrapped * fineGrid.height - 0.5;
    const rx = 2 * (px - Math.floor(px)) - 1;
    const ry = 2 * (py - Math.floor(py)) - 1;

    // 4. Normalized mip and constant bias
    const normMip = ctx.normalizedMip ?? 0;
    const constBias = 1.0;

    // 5. Construct input vector x: Fine (ch) -> Coarse (ch) -> [rx, ry, normMip, constBias]
    const x = new Float32Array(spec.inputDim);
    for (let c = 0; c < ch; c++) {
        x[c] = fine[c]!;
        x[ch + c] = coarse[c]!;
    }
    x[ch * 2 + 0] = rx;
    x[ch * 2 + 1] = ry;
    x[ch * 2 + 2] = normMip;
    x[ch * 2 + 3] = constBias;

    // 6. Forward pass through canonical pure MLP function
    return evaluateMlpStandardCpu(x, weights);
}

/**
 * Evaluates the CPU reference neural material decoder.
 * Adheres strictly to the VNM1 mathematical specifications in NTC_PLAN.md.
 */
export function decodeNeuralMaterialCpu(
    fineGrid: LatentFeatureGrid,
    coarseGrid: LatentFeatureGrid,
    weights: MlpWeights,
    ctx: SampleContext,
): DecodedPbrChannels {
    const y = decodeNeuralMaterialRawCpu(fineGrid, coarseGrid, weights, ctx);

    // Post-processing PBR channels:
    // 0..2: Base Color (linear, clamp 0..1)
    const r = Math.max(0, Math.min(1, y[0]));
    const g = Math.max(0, Math.min(1, y[1]));
    const b = Math.max(0, Math.min(1, y[2]));

    // 3..4: Tangent Normal XY (-1..1), compute Z = sqrt(max(0, 1 - x^2 - y^2))
    let nx = Math.max(-1, Math.min(1, y[3]));
    let ny = Math.max(-1, Math.min(1, y[4]));
    const xyLenSq = nx * nx + ny * ny;
    if (xyLenSq > 1.0) {
        const s = 1.0 / Math.sqrt(xyLenSq);
        nx *= s;
        ny *= s;
    }
    const nz = Math.sqrt(Math.max(0, 1.0 - (nx * nx + ny * ny)));
    const normalLen = Math.hypot(nx, ny, nz) || 1.0;

    // 5: Roughness (0..1)
    const roughness = Math.max(0, Math.min(1, y[5]));
    // 6: Metalness (0..1)
    const metalness = Math.max(0, Math.min(1, y[6]));
    // 7: AO (0..1)
    const ao = Math.max(0, Math.min(1, y[7]));

    return {
        raw: { values: [y[0]!, y[1]!, y[2]!, y[3]!, y[4]!, y[5]!, y[6]!, y[7]!] },
        baseColor: [r, g, b],
        normal: [nx / normalLen, ny / normalLen, nz / normalLen],
        roughness,
        metalness,
        ao,
    };
}
