/** Copies tightly packed or 256-byte-row-aligned RGBA f32 readback into logical pixels. */
export function unpackRgba32Readback(raw: unknown, width: number, height: number): Float32Array {
    if (![width, height].every((n) => Number.isSafeInteger(n) && n > 0 && n <= 256)) throw new Error('Invalid readback dimensions');
    if (!(raw instanceof Float32Array)) throw new Error('Readback must be Float32Array');
    const rowFloats = width * 4;
    const stride = Math.ceil(rowFloats * 4 / 256) * 64;
    const packedLength = rowFloats * height;
    const paddedLength = stride * (height - 1) + rowFloats;
    if (raw.length === packedLength) return raw.slice();
    if (raw.length !== paddedLength) throw new Error('Readback buffer size mismatch');
    const pixels = new Float32Array(packedLength);
    for (let y = 0; y < height; y++) pixels.set(raw.subarray(y * stride, y * stride + rowFloats), y * rowFloats);
    return pixels;
}
