import { MLP_PROFILES, type LatentFeatureGrid, type MlpProfileSpec, type MlpWeights } from './types';

export function validateMlpWeights(weights: MlpWeights): MlpProfileSpec {
    if (!Object.prototype.hasOwnProperty.call(MLP_PROFILES, weights.profile)) throw new Error(`Unsupported profile: ${weights.profile}`);
    const spec = MLP_PROFILES[weights.profile];
    const layers: readonly [keyof Omit<MlpWeights, 'profile'>, number][] = [
        ['w1', spec.hidden1 * spec.inputDim], ['b1', spec.hidden1],
        ['w2', spec.hidden2 * spec.hidden1], ['b2', spec.hidden2],
        ['w3', spec.outputDim * spec.hidden2], ['b3', spec.outputDim],
    ];
    for (const [name, length] of layers) {
        const values = weights[name];
        if (!(values instanceof Float32Array) || values.length !== length) {
            throw new Error(`${name} length mismatch: expected ${length} f32 values`);
        }
        if (!values.every(Number.isFinite)) throw new Error(`Non-finite weight in ${name}`);
    }
    return spec;
}

export function validateLatentDimensions(width: number, height: number): void {
    if (![width, height].every((n) => Number.isInteger(n) && n > 0 && n <= 65535)) {
        throw new Error('Latent dimensions must be positive uint16 integers');
    }
}

export function validateU8LatentGrid(grid: LatentFeatureGrid, expectedChannels?: number): number {
    validateLatentDimensions(grid.width, grid.height);
    if (grid.isNormalizedU8 !== true || !(grid.data instanceof Uint8Array)) {
        throw new Error('Only normalized u8 latent grids (Uint8Array) are supported');
    }
    if (expectedChannels !== undefined && (!Number.isInteger(expectedChannels) || expectedChannels <= 0)) {
        throw new Error(`Invalid expected latent channel count: ${expectedChannels}`);
    }
    if (grid.channels !== undefined && (!Number.isInteger(grid.channels) || grid.channels <= 0)) {
        throw new Error(`Invalid latent grid channel count: ${grid.channels}`);
    }
    if (expectedChannels !== undefined && grid.channels !== undefined && grid.channels !== expectedChannels) {
        throw new Error(`Latent grid channel count mismatch: expected ${expectedChannels}, got ${grid.channels}`);
    }
    const channels = grid.channels ?? expectedChannels ?? 4;
    if (grid.data.byteLength !== grid.width * grid.height * channels) {
        throw new Error('Latent grid byteLength mismatch');
    }
    return channels;
}
