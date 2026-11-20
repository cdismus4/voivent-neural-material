import { describe, expect, it } from 'vitest';
import type { WebGPURenderer } from 'three/webgpu';
import {
    createVerificationLatentGrids,
    createVerificationWeights,
    runGpuNeuralMaterialVerification,
} from '../gpu/GpuReadbackVerification';
import { decodeNeuralMaterialCpu, decodeNeuralMaterialRawCpu } from '../reference/CpuReferenceDecoder';

describe('VNM1 Ticket B2: GPU Readback Verification Harness', () => {
    it('generates reproducible deterministic latent grids and MLP weights', () => {
        const { fine: f1, coarse: c1 } = createVerificationLatentGrids(32, 16, 16, 8, 777);
        const { fine: f2, coarse: c2 } = createVerificationLatentGrids(32, 16, 16, 8, 777);

        expect(f1.width).toBe(32);
        expect(f1.height).toBe(16);
        expect(c1.width).toBe(16);
        expect(c1.height).toBe(8);
        expect(f1.isNormalizedU8).toBe(true);

        // Deterministic equality
        expect(Array.from(f1.data)).toEqual(Array.from(f2.data));
        expect(Array.from(c1.data)).toEqual(Array.from(c2.data));

        const w1 = createVerificationWeights(888);
        const w2 = createVerificationWeights(888);
        expect(Array.from(w1.w1)).toEqual(Array.from(w2.w1));
        expect(Array.from(w1.b3)).toEqual(Array.from(w2.b3));
    });

    it('verifies decodeNeuralMaterialRawCpu matches decodeNeuralMaterialCpu post-processing pipeline', () => {
        const { fine, coarse } = createVerificationLatentGrids(16, 16, 8, 8, 123);
        const weights = createVerificationWeights(456);

        const ctx = { u: 0.35, v: 0.72, wrapU: 'repeat' as const, wrapV: 'repeat' as const };
        const raw = decodeNeuralMaterialRawCpu(fine, coarse, weights, ctx);
        const pbr = decodeNeuralMaterialCpu(fine, coarse, weights, ctx);

        expect(raw.length).toBe(8);

        // Raw 0..2 are unclamped BaseColor linear
        expect(pbr.baseColor[0]).toBeCloseTo(Math.max(0, Math.min(1, raw[0]!)), 5);
        expect(pbr.baseColor[1]).toBeCloseTo(Math.max(0, Math.min(1, raw[1]!)), 5);
        expect(pbr.baseColor[2]).toBeCloseTo(Math.max(0, Math.min(1, raw[2]!)), 5);

        // Raw 5..7 are roughness, metalness, ao
        expect(pbr.roughness).toBeCloseTo(Math.max(0, Math.min(1, raw[5]!)), 5);
        expect(pbr.metalness).toBeCloseTo(Math.max(0, Math.min(1, raw[6]!)), 5);
        expect(pbr.ao).toBeCloseTo(Math.max(0, Math.min(1, raw[7]!)), 5);
    });

    it('strictly rejects invalid tolerance, empty test cases, and invalid resolutions', async () => {
        const dummyRenderer = {
            getRenderTarget: () => null,
            setRenderTarget: () => {},
            render: () => {},
        } as unknown as WebGPURenderer;

        // 1. Invalid tolerance
        await expect(
            runGpuNeuralMaterialVerification(dummyRenderer, { tolerance: -1.0 }),
        ).rejects.toThrow(/Invalid tolerance threshold/);

        await expect(
            runGpuNeuralMaterialVerification(dummyRenderer, { tolerance: NaN }),
        ).rejects.toThrow(/Invalid tolerance threshold/);

        // 2. Empty test cases (non-existent filter IDs)
        await expect(
            runGpuNeuralMaterialVerification(dummyRenderer, { filterCaseIds: ['non-existent-case-id'] }),
        ).rejects.toThrow(/No valid test cases selected/);

        // 3. Invalid resolution
        await expect(
            runGpuNeuralMaterialVerification(dummyRenderer, { resolution: 0 }),
        ).rejects.toThrow(/Invalid resolution/);

        await expect(
            runGpuNeuralMaterialVerification(dummyRenderer, { resolution: -4 }),
        ).rejects.toThrow(/Invalid resolution/);
    });

    it('strictly fails test and reports maxError as Infinity when readback contains NaNs', async () => {
        const nanRenderer = {
            init: async () => {},
            getActiveCubeFace: () => 0,
            getActiveMipmapLevel: () => 0,
            getRenderTarget: () => null,
            setRenderTarget: () => {},
            render: () => {},
            backend: { isWebGPUBackend: true, adapter: { info: { vendor: 'test-gpu', architecture: 'mock', device: 'test' } } },
            readRenderTargetPixelsAsync: async (_rt: unknown, _x: number, _y: number, w: number, h: number) => {
                const buf = new Float32Array(w * h * 4);
                buf.fill(NaN); // All NaNs!
                return buf;
            },
        } as unknown as WebGPURenderer;

        const report = await runGpuNeuralMaterialVerification(nanRenderer, {
            tolerance: 1e-3,
            resolution: 4,
            filterCaseIds: ['minimal-1x1-grid'],
        });

        expect(report.passed).toBe(false);
        expect(report.maxErrorAcrossAllTests).toBe(Infinity);
        expect(report.testCases[0]!.passed).toBe(false);
        expect(report.testCases[0]!.maxAbsoluteError).toBe(Infinity);
        expect(report.markdownReport).toContain('FAILED');
    });

    it('validates exact match (< 1e-4) when mock returns faithful CPU decoded pixels', async () => {
        // Custom reader that returns pass 0 and pass 1 accurately
        let callCount = 0;
        const faithfulMultiPassRenderer = {
            init: async () => {},
            getActiveCubeFace: () => 0,
            getActiveMipmapLevel: () => 0,
            getRenderTarget: () => null,
            setRenderTarget: () => {},
            render: () => {},
            backend: {
                isWebGPUBackend: true,
                adapter: {
                    info: {
                        vendor: 'apple',
                        architecture: 'metal-3',
                        device: 'Apple M4 WebGPU (Simulated)',
                    },
                },
            },
            readRenderTargetPixelsAsync: async (
                _rt: unknown,
                _x: number,
                _y: number,
                w: number,
                h: number,
            ): Promise<Float32Array> => {
                const sharedWeights = createVerificationWeights();
                const { fine, coarse } = createVerificationLatentGrids(1, 1, 1, 1);
                const buf = new Float32Array(w * h * 4);
                const isPass0 = callCount % 2 === 0;
                callCount++;

                for (let py = 0; py < h; py++) {
                    for (let px = 0; px < w; px++) {
                        const u = (px + 0.5) / w;
                        const v = (py + 0.5) / h;
                        const raw = decodeNeuralMaterialRawCpu(fine, coarse, sharedWeights, {
                            u,
                            v,
                            wrapU: 'repeat',
                            wrapV: 'repeat',
                        });
                        const idx = (py * w + px) * 4;
                        if (isPass0) {
                            buf[idx + 0] = raw[0]!;
                            buf[idx + 1] = raw[1]!;
                            buf[idx + 2] = raw[2]!;
                            buf[idx + 3] = raw[3]!;
                        } else {
                            buf[idx + 0] = raw[4]!;
                            buf[idx + 1] = raw[5]!;
                            buf[idx + 2] = raw[6]!;
                            buf[idx + 3] = raw[7]!;
                        }
                    }
                }
                return buf;
            },
        } as unknown as WebGPURenderer;

        const report = await runGpuNeuralMaterialVerification(faithfulMultiPassRenderer, {
            tolerance: 1e-4,
            resolution: 4,
            filterCaseIds: ['minimal-1x1-grid'],
        });

        expect(report.passed).toBe(true);
        expect(report.maxErrorAcrossAllTests).toBeLessThan(1e-5);
        expect(report.testCases[0]!.passed).toBe(true);
        expect(report.markdownReport).toContain('PASSED');
    });

    it('flags test failure and generates worst-sample diagnostics when GPU readback deviates from CPU', async () => {
        const mockRenderer = {
            init: async () => {},
            getActiveCubeFace: () => 0,
            getActiveMipmapLevel: () => 0,
            getRenderTarget: () => null,
            setRenderTarget: () => {},
            render: () => {},
            backend: { isWebGPUBackend: true, adapter: { info: { vendor: 'test', architecture: 'mock', device: 'test' } } },
            readRenderTargetPixelsAsync: async (_rt: unknown, _x: number, _y: number, w: number, h: number) => {
                const buf = new Float32Array(w * h * 4);
                buf.fill(99.0); // Extreme deviation to guarantee failure
                return buf;
            },
        } as unknown as WebGPURenderer;

        const report = await runGpuNeuralMaterialVerification(mockRenderer, {
            tolerance: 1e-3, // Strict tolerance
            resolution: 4,
            filterCaseIds: ['minimal-1x1-grid'],
        });

        expect(report.passed).toBe(false);
        expect(report.testCases[0]!.passed).toBe(false);
        expect(report.testCases[0]!.maxAbsoluteError).toBeGreaterThan(1.0);
        expect(report.testCases[0]!.worstSample).toBeDefined();
        expect(report.markdownReport).toContain('FAILED');
    });
});
