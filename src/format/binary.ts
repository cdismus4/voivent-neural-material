import type { DecoderProfile, LatentFeatureGrid, MlpWeights } from './types';
import { getLatentChannelCount, MLP_PROFILES } from './types';
import { validateMlpWeights, validateU8LatentGrid } from './validation';

export const VNM_MAGIC = 0x314D4E56; // 'VNM1' in little-endian ASCII
export const VNM_VERSION = 1;

export enum VnmChunkKind {
    Weights = 1,
    Latents = 2,
    Tail = 3,
}

export interface VnmHeader {
    magic: number;
    version: number;
    chunkKind: VnmChunkKind;
    payloadByteLength: number;
}

export function writeVnmHeader(chunkKind: VnmChunkKind, payloadByteLength: number): Uint8Array {
    if (!Number.isInteger(payloadByteLength) || payloadByteLength < 0 || payloadByteLength > 0xffffffff) {
        throw new Error('Payload byte length must fit an unsigned 32-bit integer');
    }
    if (chunkKind !== VnmChunkKind.Weights && chunkKind !== VnmChunkKind.Latents && chunkKind !== VnmChunkKind.Tail) {
        throw new Error(`Invalid VNM chunk kind: ${chunkKind}`);
    }
    const buffer = new ArrayBuffer(16);
    const view = new DataView(buffer);
    view.setUint32(0, VNM_MAGIC, true);
    view.setUint16(4, VNM_VERSION, true);
    view.setUint16(6, chunkKind, true);
    view.setUint32(8, payloadByteLength, true);
    view.setUint32(12, 0, true); // reserved
    return new Uint8Array(buffer);
}

export function parseVnmHeader(data: Uint8Array): VnmHeader {
    if (data.byteLength < 16) {
        throw new Error(`VNM chunk too small: expected at least 16 bytes header, got ${data.byteLength}`);
    }
    const view = new DataView(data.buffer, data.byteOffset, 16);
    const magic = view.getUint32(0, true);
    if (magic !== VNM_MAGIC) {
        throw new Error(`Invalid VNM magic: expected 0x${VNM_MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
    }
    const version = view.getUint16(4, true);
    if (version !== VNM_VERSION) {
        throw new Error(`Unsupported VNM version: expected ${VNM_VERSION}, got ${version}`);
    }
    const chunkKindRaw = view.getUint16(6, true);
    if (chunkKindRaw !== VnmChunkKind.Weights && chunkKindRaw !== VnmChunkKind.Latents && chunkKindRaw !== VnmChunkKind.Tail) {
        throw new Error(`Invalid or unknown VNM chunk kind: ${chunkKindRaw}`);
    }
    const chunkKind = chunkKindRaw as VnmChunkKind;
    const payloadByteLength = view.getUint32(8, true);
    const reserved = view.getUint32(12, true);

    if (reserved !== 0) {
        throw new Error(`Invalid VNM header reserved field: expected 0, got ${reserved}`);
    }

    if (data.byteLength !== 16 + payloadByteLength) {
        throw new Error(
            `VNM chunk byte length mismatch: header specifies 16 + ${payloadByteLength} = ${16 + payloadByteLength} bytes, but received buffer of ${data.byteLength} bytes`,
        );
    }

    return { magic, version, chunkKind, payloadByteLength };
}

/**
 * Packs MLP weights into a VNM1 weights chunk (binary f32 array with strict layer dimension validation).
 */
export function serializeWeightsChunk(weights: MlpWeights): Uint8Array {
    const spec = validateMlpWeights(weights);

    const totalFloats = spec.totalWeights;
    const payloadBytes = totalFloats * 4;
    const header = writeVnmHeader(VnmChunkKind.Weights, payloadBytes);

    const out = new Uint8Array(16 + payloadBytes);
    out.set(header, 0);

    const view = new DataView(out.buffer, out.byteOffset + 16, payloadBytes);
    let byteOffset = 0;

    const writeArray = (arr: Float32Array) => {
        for (let i = 0; i < arr.length; i++) {
            view.setFloat32(byteOffset, arr[i]!, true);
            byteOffset += 4;
        }
    };

    writeArray(weights.w1);
    writeArray(weights.b1);
    writeArray(weights.w2);
    writeArray(weights.b2);
    writeArray(weights.w3);
    writeArray(weights.b3);

    return out;
}

/**
 * Parses MLP weights from a VNM1 weights chunk with full alignment safety.
 */
export function parseWeightsChunk(data: Uint8Array, profile: DecoderProfile): MlpWeights {
    const header = parseVnmHeader(data);
    if (header.chunkKind !== VnmChunkKind.Weights) {
        throw new Error(`Expected Weights chunk (${VnmChunkKind.Weights}), got ${header.chunkKind}`);
    }

    const spec = MLP_PROFILES[profile];
    if (!spec) {
        throw new Error(`Unknown MLP profile: ${profile}`);
    }

    const expectedBytes = spec.totalWeights * 4;
    if (header.payloadByteLength !== expectedBytes) {
        throw new Error(`Weights size mismatch for profile '${profile}': expected ${expectedBytes} bytes, got ${header.payloadByteLength}`);
    }

    const view = new DataView(data.buffer, data.byteOffset + 16, header.payloadByteLength);
    const readArray = (length: number, startByte: number): Float32Array => {
        const out = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            const val = view.getFloat32(startByte + i * 4, true);
            if (!Number.isFinite(val)) {
                throw new Error(`Non-finite weight detected at float index ${i}`);
            }
            out[i] = val;
        }
        return out;
    };

    let byteOffset = 0;
    const w1 = readArray(spec.hidden1 * spec.inputDim, byteOffset); byteOffset += w1.length * 4;
    const b1 = readArray(spec.hidden1, byteOffset); byteOffset += b1.length * 4;
    const w2 = readArray(spec.hidden2 * spec.hidden1, byteOffset); byteOffset += w2.length * 4;
    const b2 = readArray(spec.hidden2, byteOffset); byteOffset += b2.length * 4;
    const w3 = readArray(spec.outputDim * spec.hidden2, byteOffset); byteOffset += w3.length * 4;
    const b3 = readArray(spec.outputDim, byteOffset);

    return { profile, w1, b1, w2, b2, w3, b3 };
}

/**
 * Serializes Fine and Coarse latent feature grids into level-N.bin.
 * Enforces normalized u8 encoding for VNM1; channel count derives from profile when provided (default 4).
 * Subheader reserved bytes 10..15 remain strictly 0 per REVISION_DECISION.md.
 */
export function serializeLatentsChunk(fine: LatentFeatureGrid, coarse: LatentFeatureGrid, profile?: DecoderProfile): Uint8Array {
    const expectedCh = profile !== undefined ? getLatentChannelCount(profile) : undefined;
    const fineChannels = validateU8LatentGrid(fine, expectedCh);
    const coarseChannels = validateU8LatentGrid(coarse, expectedCh);
    if (fineChannels !== coarseChannels) {
        throw new Error(`Latent grid channel count mismatch between Fine (${fineChannels}) and Coarse (${coarseChannels})`);
    }
    if (!fine.isNormalizedU8 || !(fine.data instanceof Uint8Array)) {
        throw new Error('Only normalized u8 latent grids (Uint8Array) are supported in VNM1 level chunks');
    }
    if (!coarse.isNormalizedU8 || !(coarse.data instanceof Uint8Array)) {
        throw new Error('Only normalized u8 latent grids (Uint8Array) are supported in VNM1 level chunks');
    }
    if (fine.width < 1 || fine.height < 1) throw new Error(`Invalid fine dimensions: ${fine.width}x${fine.height}`);
    if (coarse.width < 1 || coarse.height < 1) throw new Error(`Invalid coarse dimensions: ${coarse.width}x${coarse.height}`);

    const fineExpectedBytes = fine.width * fine.height * fineChannels;
    const coarseExpectedBytes = coarse.width * coarse.height * coarseChannels;

    if (fine.data.byteLength !== fineExpectedBytes) {
        throw new Error(`Fine grid byteLength mismatch: expected ${fineExpectedBytes}, got ${fine.data.byteLength}`);
    }
    if (coarse.data.byteLength !== coarseExpectedBytes) {
        throw new Error(`Coarse grid byteLength mismatch: expected ${coarseExpectedBytes}, got ${coarse.data.byteLength}`);
    }

    const subHeaderBytes = 16;
    const payloadBytes = subHeaderBytes + fineExpectedBytes + coarseExpectedBytes;

    const header = writeVnmHeader(VnmChunkKind.Latents, payloadBytes);
    const out = new Uint8Array(16 + payloadBytes);
    out.set(header, 0);

    const view = new DataView(out.buffer, out.byteOffset + 16, subHeaderBytes);
    view.setUint16(0, fine.width, true);
    view.setUint16(2, fine.height, true);
    view.setUint16(4, coarse.width, true);
    view.setUint16(6, coarse.height, true);
    view.setUint8(8, 1); // 1 = u8 normalized
    view.setUint8(9, 1); // 1 = u8 normalized
    // reserved bytes 10..15 zeroed automatically

    let offset = 16 + subHeaderBytes;
    out.set(new Uint8Array(fine.data.buffer, fine.data.byteOffset, fineExpectedBytes), offset);
    offset += fineExpectedBytes;
    out.set(new Uint8Array(coarse.data.buffer, coarse.data.byteOffset, coarseExpectedBytes), offset);

    return out;
}

/**
 * Parses Fine and Coarse latent feature grids from level-N.bin with full payload size validation.
 * Channel count is derived strictly from the profile (default 4); reserved subheader bytes 10..15 must be 0.
 */
export function parseLatentsChunk(
    data: Uint8Array,
    profileOrChannels?: DecoderProfile | number,
): { fine: LatentFeatureGrid; coarse: LatentFeatureGrid } {
    let channels: number;
    if (profileOrChannels === undefined) {
        channels = 4;
    } else if (typeof profileOrChannels === 'number') {
        if (!Number.isInteger(profileOrChannels) || profileOrChannels <= 0) {
            throw new Error(`Invalid latent channel count: ${profileOrChannels}`);
        }
        channels = profileOrChannels;
    } else {
        channels = getLatentChannelCount(profileOrChannels);
    }

    const header = parseVnmHeader(data);
    if (header.chunkKind !== VnmChunkKind.Latents) {
        throw new Error(`Expected Latents chunk (${VnmChunkKind.Latents}), got ${header.chunkKind}`);
    }

    const subHeaderBytes = 16;
    if (header.payloadByteLength < subHeaderBytes) {
        throw new Error(`Latent chunk payload too small for subheader: ${header.payloadByteLength} < 16`);
    }

    const view = new DataView(data.buffer, data.byteOffset + 16, subHeaderBytes);
    const fineWidth = view.getUint16(0, true);
    const fineHeight = view.getUint16(2, true);
    const coarseWidth = view.getUint16(4, true);
    const coarseHeight = view.getUint16(6, true);
    const fineFlag = view.getUint8(8);
    const coarseFlag = view.getUint8(9);

    if (fineFlag !== 1 || coarseFlag !== 1) {
        throw new Error(`Unsupported latent quantization flag: expected u8 (1), got Fine(${fineFlag}) Coarse(${coarseFlag})`);
    }

    for (let b = 10; b < 16; b++) {
        if (view.getUint8(b) !== 0) {
            throw new Error(`Non-zero reserved byte in latent subheader at offset ${b}`);
        }
    }

    if (fineWidth < 1 || fineHeight < 1 || coarseWidth < 1 || coarseHeight < 1) {
        throw new Error(`Invalid dimensions in latent header: Fine(${fineWidth}x${fineHeight}), Coarse(${coarseWidth}x${coarseHeight})`);
    }

    const fineBytes = fineWidth * fineHeight * channels;
    const coarseBytes = coarseWidth * coarseHeight * channels;
    const expectedPayload = subHeaderBytes + fineBytes + coarseBytes;

    if (header.payloadByteLength !== expectedPayload) {
        throw new Error(`Latent payload byte length mismatch: header specifies ${header.payloadByteLength}, but dimensions require ${expectedPayload}`);
    }

    let offset = 16 + subHeaderBytes;
    // Copy bytes into independent Uint8Array to prevent retaining references to outer chunk buffer
    const fineData = new Uint8Array(fineBytes);
    fineData.set(data.subarray(offset, offset + fineBytes));
    offset += fineBytes;

    const coarseData = new Uint8Array(coarseBytes);
    coarseData.set(data.subarray(offset, offset + coarseBytes));

    return {
        fine: { width: fineWidth, height: fineHeight, channels, data: fineData, isNormalizedU8: true },
        coarse: { width: coarseWidth, height: coarseHeight, channels, data: coarseData, isNormalizedU8: true },
    };
}

export interface ParsedTailChunk {
    readonly header: VnmHeader;
    readonly tailWidth: number;
    readonly tailHeight: number;
    readonly tailFirstMip: number;
    readonly channels: number;
    readonly format: number;
    readonly pixelData: Uint8Array;
}

/**
 * Parses and strictly validates a VNM1 Tail preview chunk.
 */
export function parseTailChunk(data: Uint8Array): ParsedTailChunk {
    const header = parseVnmHeader(data);
    if (header.chunkKind !== VnmChunkKind.Tail) {
        throw new Error(`Expected Tail chunk (${VnmChunkKind.Tail}), got ${header.chunkKind}`);
    }

    const subHeaderBytes = 16;
    if (header.payloadByteLength < subHeaderBytes) {
        throw new Error(`Tail chunk payload too small for subheader: ${header.payloadByteLength} < 16`);
    }

    const view = new DataView(data.buffer, data.byteOffset + 16, subHeaderBytes);
    const tailWidth = view.getUint16(0, true);
    const tailHeight = view.getUint16(2, true);
    const tailFirstMip = view.getUint8(4);
    const channels = view.getUint8(5);
    const format = view.getUint8(6);

    if (tailWidth < 1 || tailHeight < 1) {
        throw new Error(`Invalid Tail dimensions: ${tailWidth}x${tailHeight}`);
    }
    if (channels !== 8) {
        throw new Error(`Unsupported Tail channels: expected 8, got ${channels}`);
    }
    if (format !== 1) {
        throw new Error(`Unsupported Tail format flag: expected 1 (u8 unorm), got ${format}`);
    }

    for (let b = 7; b < 16; b++) {
        if (view.getUint8(b) !== 0) {
            throw new Error(`Non-zero reserved byte in tail subheader at offset ${b}`);
        }
    }

    const expectedPixelBytes = tailWidth * tailHeight * channels;
    const expectedPayload = subHeaderBytes + expectedPixelBytes;
    if (header.payloadByteLength !== expectedPayload) {
        throw new Error(
            `Tail payload byte length mismatch: header specifies ${header.payloadByteLength}, but dimensions require ${expectedPayload}`,
        );
    }

    const pixelData = new Uint8Array(expectedPixelBytes);
    pixelData.set(data.subarray(16 + subHeaderBytes, 16 + expectedPayload));

    return {
        header,
        tailWidth,
        tailHeight,
        tailFirstMip,
        channels,
        format,
        pixelData,
    };
}
