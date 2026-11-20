import { parseLatentsChunk, parseWeightsChunk } from '../format/binary';
import type { ExportedVnmPackage } from '../exporter/VnmExporter';
import { decodeNeuralMaterialRawCpu } from '../reference/CpuReferenceDecoder';
import { reconstructNormal } from './LossFunctions';
import type { PbrTextureData } from './SyntheticTextureGenerator';

export interface PbrQualityAudit {
    readonly sampleCount: number;
    readonly colorMae: number;
    readonly colorPsnr: number;
    readonly normalAngularMae: number;
    readonly roughnessMae: number;
    readonly metalnessMae: number;
    readonly aoMae: number;
    readonly overallMae: number;
    readonly maxAbsError: number;
    readonly accepted: boolean;
    readonly thresholds: {
        readonly colorMae: number;
        readonly normalAngularMae: number;
        readonly scalarMae: number;
        readonly maxAbsError: number;
    };
}

const DEFAULT_THRESHOLDS = {
    colorMae: 0.08,
    normalAngularMae: 0.16,
    scalarMae: 0.08,
    maxAbsError: 0.4,
} as const;

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

/**
 * Audits the quantized VNM1 package, not the pre-quantization trainer state.
 * This catches white/flat reconstruction failures before the asset is accepted.
 */
export function auditVnmPackageQuality(
    source: PbrTextureData,
    pkg: ExportedVnmPackage,
    sampleResolution = 32,
): PbrQualityAudit {
    const weights = parseWeightsChunk(pkg.weightsChunk.data, pkg.manifest.profile);
    const latents = parseLatentsChunk(pkg.level0Chunk.data, pkg.manifest.profile);
    const width = Math.max(1, Math.min(sampleResolution, source.width));
    const height = Math.max(1, Math.min(sampleResolution, source.height));
    const sampleCount = width * height;
    let colorAbs = 0;
    let colorSq = 0;
    let normalAngular = 0;
    let roughAbs = 0;
    let metalAbs = 0;
    let aoAbs = 0;
    let overallAbs = 0;
    let maxAbsError = 0;

    for (let y = 0; y < height; y += 1) {
        const sourceY = Math.min(source.height - 1, Math.floor(((y + 0.5) / height) * source.height));
        for (let x = 0; x < width; x += 1) {
            const sourceX = Math.min(source.width - 1, Math.floor(((x + 0.5) / width) * source.width));
            const sourceOffset = (sourceY * source.width + sourceX) * 8;
            const decoded = decodeNeuralMaterialRawCpu(latents.fine, latents.coarse, weights, {
                u: (x + 0.5) / width,
                v: (y + 0.5) / height,
                wrapU: 'repeat',
                wrapV: 'repeat',
            });

            for (let channel = 0; channel < 3; channel += 1) {
                const error = Math.abs(decoded[channel] - source.data[sourceOffset + channel]);
                colorAbs += error;
                colorSq += error * error;
                overallAbs += error;
                maxAbsError = Math.max(maxAbsError, error);
            }

            const sourceNormal = reconstructNormal(source.data[sourceOffset + 3], source.data[sourceOffset + 4]);
            const decodedNormal = reconstructNormal(decoded[3], decoded[4]);
            const dot = clamp(
                sourceNormal[0] * decodedNormal[0] + sourceNormal[1] * decodedNormal[1] + sourceNormal[2] * decodedNormal[2],
                -1,
                1,
            );
            normalAngular += Math.acos(dot);
            for (let channel = 3; channel < 5; channel += 1) {
                const error = Math.abs(decoded[channel] - source.data[sourceOffset + channel]);
                overallAbs += error;
                maxAbsError = Math.max(maxAbsError, error);
            }

            const scalarErrors = [
                [decoded[5], source.data[sourceOffset + 5]],
                [decoded[6], source.data[sourceOffset + 6]],
                [decoded[7], source.data[sourceOffset + 7]],
            ] as const;
            scalarErrors.forEach(([decodedValue, sourceValue], index) => {
                const error = Math.abs(decodedValue - sourceValue);
                if (index === 0) roughAbs += error;
                if (index === 1) metalAbs += error;
                if (index === 2) aoAbs += error;
                overallAbs += error;
                maxAbsError = Math.max(maxAbsError, error);
            });
        }
    }

    const colorMae = colorAbs / (sampleCount * 3);
    const colorRmse = Math.sqrt(colorSq / (sampleCount * 3));
    const normalAngularMae = normalAngular / sampleCount;
    const roughnessMae = roughAbs / sampleCount;
    const metalnessMae = metalAbs / sampleCount;
    const aoMae = aoAbs / sampleCount;
    const overallMae = overallAbs / (sampleCount * 8);
    const thresholds = DEFAULT_THRESHOLDS;
    const accepted = colorMae <= thresholds.colorMae
        && normalAngularMae <= thresholds.normalAngularMae
        && roughnessMae <= thresholds.scalarMae
        && metalnessMae <= thresholds.scalarMae
        && aoMae <= thresholds.scalarMae
        && maxAbsError <= thresholds.maxAbsError;

    return {
        sampleCount,
        colorMae,
        colorPsnr: -20 * Math.log10(Math.max(1e-6, colorRmse)),
        normalAngularMae,
        roughnessMae,
        metalnessMae,
        aoMae,
        overallMae,
        maxAbsError,
        accepted,
        thresholds,
    };
}
