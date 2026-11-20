import { describe, expect, it } from 'vitest';
import { RenderTarget, type WebGPURenderer } from 'three/webgpu';
import { unpackRgba32Readback } from '../gpu/ReadbackLayout';
import { runGpuNeuralMaterialVerification } from '../gpu/GpuReadbackVerification';

describe('readback layout and lifecycle', () => {
    it.each([1, 4, 8, 16, 17])('unpacks width %s without treating row padding as pixels', (width) => {
        const height = 3, row = width * 4, stride = Math.ceil(row / 64) * 64;
        const padded = new Float32Array(stride * (height - 1) + row).fill(NaN);
        for (let y = 0; y < height; y++) for (let x = 0; x < row; x++) padded[y * stride + x] = y * 1000 + x - 12;
        const packed = unpackRgba32Readback(padded, width, height);
        expect(packed.length).toBe(row * height);
        for (let y = 0; y < height; y++) expect(packed[y * row]).toBe(y * 1000 - 12);
        expect(packed.every(Number.isFinite)).toBe(true);
    });

    it('rejects missing pixels and wrong element types', () => {
        expect(() => unpackRgba32Readback(new Float32Array(3), 1, 1)).toThrow();
        expect(() => unpackRgba32Readback(new Uint32Array(4), 1, 1)).toThrow();
    });

    it('rejects non-WebGPU fallback after initialization', async () => {
        const mock = { init: async () => {}, backend: { isWebGPUBackend: false } } as unknown as WebGPURenderer;
        await expect(runGpuNeuralMaterialVerification(mock)).rejects.toThrow(/requires the WebGPU/);
    });

    it.each([false, true])('restores caller target, face and mip before yielding (render failure: %s)', async (failRender) => {
        const original = new RenderTarget();
        let current: RenderTarget | null = original, face = 3, mip = 2;
        let rendered = false, disposedTargets = 0;
        const seen = new Set<RenderTarget>();
        const mock = {
            init: async () => {}, backend: { isWebGPUBackend: true },
            getRenderTarget: () => current,
            getActiveCubeFace: () => face, getActiveMipmapLevel: () => mip,
            setRenderTarget: (target: RenderTarget | null, nextFace = 0, nextMip = 0) => {
                current = target; face = nextFace; mip = nextMip;
                if (target && target !== original && !seen.has(target)) {
                    seen.add(target); target.addEventListener('dispose', () => { disposedTargets++; });
                }
            },
            render: () => { rendered = true; if (failRender) throw new Error('render failed'); },
            readRenderTargetPixelsAsync: async (_rt: unknown, _x: number, _y: number, w: number, h: number) => {
                expect(current).toBe(original); expect(face).toBe(3); expect(mip).toBe(2);
                return new Float32Array(w * h * 4).fill(NaN);
            },
        } as unknown as WebGPURenderer;
        try {
            const result = runGpuNeuralMaterialVerification(mock, { resolution: 1, filterCaseIds: ['minimal-1x1-grid'] });
            if (failRender) await expect(result).rejects.toThrow('render failed');
            else {
                const report = await result;
                expect(report.evidence).toBe('simulation');
                expect(report.passed).toBe(false);
                expect(report.testCases[0].worstSample?.maxDiff).toBe(Infinity);
                expect(report.markdownReport).toContain('simulation');
            }
            expect(rendered).toBe(true);
            expect(current).toBe(original); expect(face).toBe(3); expect(mip).toBe(2);
            expect(disposedTargets).toBe(seen.size);
        } finally { original.dispose(); }
    });
});
