import type { PbrTextureData } from './SyntheticTextureGenerator';

export function validatePbrTexture(source: PbrTextureData): void {
    if (![source.width, source.height].every((n) => Number.isSafeInteger(n) && n > 0) ||
        !Number.isSafeInteger(source.width * source.height * 8) ||
        !(source.data instanceof Float32Array) || source.data.length !== source.width * source.height * 8) {
        throw new Error('Invalid PBR texture dimensions or data length');
    }
    for (let i = 0; i < source.data.length; i++) {
        const value = source.data[i];
        const isNormal = i % 8 === 3 || i % 8 === 4;
        if (!Number.isFinite(value) || value < (isNormal ? -1 : 0) || value > 1) {
            throw new Error(`Invalid or non-finite PBR channel at ${i}`);
        }
    }
}
