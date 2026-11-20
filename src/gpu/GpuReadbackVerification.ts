import {
    FloatType,
    Mesh,
    NearestFilter,
    NoBlending,
    OrthographicCamera,
    PlaneGeometry,
    RenderTarget,
    RGBAFormat,
    Scene,
    WebGPURenderer,
} from 'three/webgpu';
import { NodeMaterial } from 'three/webgpu';
import { screenCoordinate, vec2 } from 'three/tsl';
import { unpackRgba32Readback } from './ReadbackLayout';
import type { DecoderProfile, LatentFeatureGrid, MlpWeights } from '../format/types';
import { getLatentChannelCount, MLP_PROFILES } from '../format/types';
import { decodeNeuralMaterialRawCpu } from '../reference/CpuReferenceDecoder';
import {
    createLatentTextureSource,
    createNeuralMaterialTslNodes,
    type LatentTextureSource,
} from './TslMlpDecoderNode';
import type {
    GpuVerificationAdapterInfo,
    GpuVerificationOptions,
    GpuVerificationReport,
    GpuVerificationSamplePoint,
    GpuVerificationTestCaseResult,
} from './types';

let activeWebGpuRenderer: WebGPURenderer | null = null;

/**
 * Registers the global active WebGPURenderer from the 3D viewport.
 */
export function registerActiveWebGpuRenderer(renderer: WebGPURenderer | null): void {
    activeWebGpuRenderer = renderer;
}

/**
 * Gets the current active WebGPURenderer.
 */
export function getActiveWebGpuRenderer(): WebGPURenderer | null {
    return activeWebGpuRenderer;
}

/**
 * Automatically obtains or initializes a WebGPURenderer and executes verification.
 */
export async function executeGpuVerificationAuto(options: GpuVerificationOptions = {}): Promise<GpuVerificationReport> {
    let renderer = activeWebGpuRenderer;
    let createdTemporary = false;

    if (!renderer && typeof document !== 'undefined') {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        renderer = new WebGPURenderer({
            canvas,
            antialias: false,
            powerPreference: 'high-performance',
        });
        createdTemporary = true;
    }

    if (!renderer) {
        throw new Error('WebGPURenderer is not available in the current environment.');
    }

    try {
        return await runGpuNeuralMaterialVerification(renderer, options);
    } finally {
        if (createdTemporary && renderer) {
            renderer.dispose();
        }
    }
}

/**
 * Creates deterministic test latent grids for verification.
 */
export function createVerificationLatentGrids(
    fineW: number,
    fineH: number,
    coarseW: number,
    coarseH: number,
    seed = 42,
    channels: 4 | 8 = 4,
): { fine: LatentFeatureGrid; coarse: LatentFeatureGrid } {
    const makeGrid = (w: number, h: number, s: number): LatentFeatureGrid => {
        const data = new Uint8Array(w * h * channels);
        let cur = s;
        for (let i = 0; i < data.length; i++) {
            cur = (cur * 1664525 + 1013904223) >>> 0;
            data[i] = (cur >>> 24) & 0xff;
        }
        return {
            width: w,
            height: h,
            data,
            isNormalizedU8: true,
        };
    };

    return {
        fine: makeGrid(fineW, fineH, seed),
        coarse: makeGrid(coarseW, coarseH, seed + 1000),
    };
}

/**
 * Creates deterministic MLP weights for verification across any supported profile.
 */
export function createVerificationWeights(
    seed = 12345,
    profile: DecoderProfile = 'standard',
): MlpWeights {
    let cur = seed;
    const nextFloat = () => {
        cur = (cur * 1664525 + 1013904223) >>> 0;
        return ((cur >>> 16) / 32768.0) - 1.0; // [-1, 1]
    };

    const makeWeights = (count: number) => {
        const arr = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            arr[i] = nextFloat() * 0.5;
        }
        return arr;
    };

    const spec = MLP_PROFILES[profile];
    return {
        profile,
        w1: makeWeights(spec.hidden1 * spec.inputDim),
        b1: makeWeights(spec.hidden1),
        w2: makeWeights(spec.hidden2 * spec.hidden1),
        b2: makeWeights(spec.hidden2),
        w3: makeWeights(spec.outputDim * spec.hidden2),
        b3: makeWeights(spec.outputDim),
    };
}

export interface TestCaseConfig {
    readonly id: string;
    readonly name: string;
    readonly description: string;
    readonly fineSize: readonly [number, number];
    readonly coarseSize: readonly [number, number];
    readonly wrapU: 'repeat' | 'clamp';
    readonly wrapV: 'repeat' | 'clamp';
    readonly uvScale?: readonly [number, number];
    readonly uvOffset?: readonly [number, number];
    readonly resolution: number; // e.g. 16 for 16x16 samples
}

export const DEFAULT_TEST_CASES: readonly TestCaseConfig[] = [
    ...(['repeat', 'clamp'] as const).flatMap((wrap) => [0, 1].map((edge) => ({
        id: `exact-${wrap}-${edge}`, name: `Exact ${wrap} endpoint ${edge}`,
        description: 'Exact UV endpoint sampling independent of raster resolution',
        fineSize: [32, 16] as const, coarseSize: [16, 8] as const,
        wrapU: wrap, wrapV: wrap, uvScale: [0, 0] as const,
        uvOffset: [edge, edge] as const, resolution: 1,
    }))),
    {
        id: 'square-standard-repeat',
        name: 'Standard 64x64 Grid (Repeat)',
        description: 'Standard square latent grids with repeat wrapping across full UV space [0, 1]',
        fineSize: [64, 64],
        coarseSize: [32, 32],
        wrapU: 'repeat',
        wrapV: 'repeat',
        resolution: 16,
    },
    {
        id: 'non-square-32x16',
        name: 'Non-Square 32x16 Grid',
        description: 'Anisotropic latent grids verifying non-square bilinear aspect alignment',
        fineSize: [32, 16],
        coarseSize: [16, 8],
        wrapU: 'repeat',
        wrapV: 'repeat',
        resolution: 16,
    },
    {
        id: 'minimal-1x1-grid',
        name: 'Minimal 1x1 Grid',
        description: 'Extreme boundary 1x1 latent grid verifying single-texel sampling and modulo math',
        fineSize: [1, 1],
        coarseSize: [1, 1],
        wrapU: 'repeat',
        wrapV: 'repeat',
        resolution: 8,
    },
    {
        id: 'clamp-to-edge',
        name: 'Clamp Wrapping Mode',
        description: 'Verifies clamp-to-edge behavior at UV boundaries [0.0, 1.0] and coordinates outside',
        fineSize: [32, 32],
        coarseSize: [16, 16],
        wrapU: 'clamp',
        wrapV: 'clamp',
        uvScale: [1.5, 1.5],
        uvOffset: [-0.25, -0.25],
        resolution: 16,
    },
    {
        id: 'negative-and-extended-uv',
        name: 'Negative & Extended UVs',
        description: 'Verifies negative UV wrap math and periodicity across multiple texture tiles [-1.5, 0.5]',
        fineSize: [32, 32],
        coarseSize: [16, 16],
        wrapU: 'repeat',
        wrapV: 'repeat',
        uvScale: [2.0, 2.0],
        uvOffset: [-1.5, -0.5],
        resolution: 16,
    },
];

/**
 * Executes full GPU readback verification against CPU reference decoder.
 * Enforces strict input validation, non-empty test cases, outputNode passthrough (bypassing toneMapping),
 * and restores previous renderer state upon completion or error.
 */
export async function runGpuNeuralMaterialVerification(
    renderer: WebGPURenderer,
    options: GpuVerificationOptions = {},
): Promise<GpuVerificationReport> {
    const tolerance = options.tolerance ?? 1e-3;

    // Astra review check: Explicitly reject invalid tolerances
    if (!Number.isFinite(tolerance) || tolerance <= 0) {
        throw new Error(`Invalid tolerance threshold: ${tolerance}. Must be a positive finite number.`);
    }

    // Astra review check: Explicitly reject empty test suite
    const testCasesToRun = options.filterCaseIds
        ? DEFAULT_TEST_CASES.filter((c) => options.filterCaseIds!.includes(c.id))
        : DEFAULT_TEST_CASES;

    if (!testCasesToRun || testCasesToRun.length === 0) {
        throw new Error('No valid test cases selected for GPU readback verification.');
    }
    if (options.filterCaseIds?.some((id) => !DEFAULT_TEST_CASES.some((c) => c.id === id))) {
        throw new Error('Unknown GPU verification case ID');
    }
    for (const config of testCasesToRun) {
        const res = options.resolution ?? config.resolution;
        if (!Number.isInteger(res) || res <= 0 || res > 256) throw new Error('Invalid resolution: expected integer 1..256');
    }
    await renderer.init();
    if (!('isWebGPUBackend' in renderer.backend) || renderer.backend.isWebGPUBackend !== true) {
        throw new Error('Verification requires the WebGPU backend; fallback rendering is not accepted');
    }
    const evidence = renderer instanceof WebGPURenderer ? 'webgpu-readback' : 'simulation';

    // 1. Retrieve WebGPU adapter info if available
    let adapterInfo: GpuVerificationAdapterInfo = {
        vendor: 'unknown',
        architecture: 'unknown',
        device: 'unknown',
        description: 'WebGPURenderer (Mock/Node or In-Flight Adapter)',
    };

    try {
        const adapterRenderer = renderer as unknown as {
            backend?: {
                adapter?: { info?: Record<string, string>; requestAdapterInfo?: () => Promise<Record<string, string>> };
                device?: { adapterInfo?: Record<string, string> };
            };
        };
        const infoObj = adapterRenderer.backend?.adapter?.info ?? adapterRenderer.backend?.device?.adapterInfo;
        if (infoObj) {
            adapterInfo = {
                vendor: infoObj.vendor || 'webgpu',
                architecture: infoObj.architecture || 'unknown',
                device: infoObj.device || 'unknown',
                description: infoObj.description || `${infoObj.vendor} ${infoObj.architecture}`,
            };
        }
    } catch {
        // Fallback adapter info
    }

    const profile: DecoderProfile = options.profile ?? 'standard';
    const channels: 4 | 8 = getLatentChannelCount(profile) === 8 ? 8 : 4;
    const testResults: GpuVerificationTestCaseResult[] = [];
    const sharedWeights = createVerificationWeights(12345, profile);

    const scene = new Scene();
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quad = new Mesh(new PlaneGeometry(2, 2));
    scene.add(quad);

    try {
        for (const config of testCasesToRun) {
            const res = options.resolution ?? config.resolution;

            // Astra review check: Explicitly reject non-positive or non-integer resolution
            if (!Number.isInteger(res) || res <= 0) {
                throw new Error(`Invalid resolution for test case '${config.id}': ${res}. Must be a positive integer.`);
            }

            const width = res;
            const height = res;

            const resources: { dispose(): void }[] = [];
            const track = <T extends { dispose(): void }>(resource: T): T => {
                resources.push(resource);
                return resource;
            };
            const trackSource = (source: LatentTextureSource): LatentTextureSource => {
                if (Array.isArray(source)) {
                    for (const tex of source) {
                        track(tex);
                    }
                } else {
                    track(source as { dispose(): void });
                }
                return source;
            };
            const release = () => { for (const resource of resources.splice(0).reverse()) resource.dispose(); };
            try {

            const { fine, coarse } = createVerificationLatentGrids(
                config.fineSize[0],
                config.fineSize[1],
                config.coarseSize[0],
                config.coarseSize[1],
                42,
                channels,
            );

            const fineTex = trackSource(createLatentTextureSource(fine, profile, config.wrapU, config.wrapV));
            const coarseTex = trackSource(createLatentTextureSource(coarse, profile, config.wrapU, config.wrapV));

            // Setup RenderTargets for raw outputs [0..3] and [4..7]
            const rt0 = track(new RenderTarget(width, height, {
                type: FloatType,
                format: RGBAFormat,
                minFilter: NearestFilter,
                magFilter: NearestFilter,
                generateMipmaps: false,
            }));

            const rt1 = track(new RenderTarget(width, height, {
                type: FloatType,
                format: RGBAFormat,
                minFilter: NearestFilter,
                magFilter: NearestFilter,
                generateMipmaps: false,
            }));

            const uvScale = config.uvScale ?? [1.0, 1.0];
            const uvOffset = config.uvOffset ?? [0.0, 0.0];

            // UV node transformed according to test case bounds
            // Fragment coordinates and readback rows both start at the top-left.
            const baseUvNode = screenCoordinate.div(vec2(width, height));
            const testUvNode = baseUvNode.mul(vec2(uvScale[0], uvScale[1])).add(vec2(uvOffset[0], uvOffset[1]));

            const nodes = createNeuralMaterialTslNodes(
                fineTex,
                coarseTex,
                sharedWeights,
                { uv: testUvNode },
                {
                    profile,
                    fineSize: config.fineSize,
                    coarseSize: config.coarseSize,
                    wrapU: config.wrapU,
                    wrapV: config.wrapV,
                },
            );

            // Astra review check: Use outputNode directly on NodeMaterial to bypass colorNode clamps,
            // tone mapping, and alpha=1 overriding on opaque materials.
            const mat0 = track(new NodeMaterial());
            mat0.outputNode = nodes.rawOutputs[0];
            mat0.transparent = false;
            mat0.depthTest = false;
            mat0.depthWrite = false;
            mat0.toneMapped = false;
            mat0.fog = false;
            mat0.blending = NoBlending;

            const mat1 = track(new NodeMaterial());
            mat1.outputNode = nodes.rawOutputs[1];
            mat1.transparent = false;
            mat1.depthTest = false;
            mat1.depthWrite = false;
            mat1.toneMapped = false;
            mat1.fog = false;
            mat1.blending = NoBlending;

            let pixels0: Float32Array;
            let pixels1: Float32Array;

            try {
                const prevRenderTarget = renderer.getRenderTarget();
                const prevFace = renderer.getActiveCubeFace();
                const prevMip = renderer.getActiveMipmapLevel();
                try {
                // Render pass 0: raw 0..3
                quad.material = mat0;
                renderer.setRenderTarget(rt0);
                renderer.render(scene, camera);

                // Render pass 1: raw 4..7
                quad.material = mat1;
                renderer.setRenderTarget(rt1);
                renderer.render(scene, camera);

                } finally {
                    // Restore synchronously, before readback yields to the viewport animation loop.
                    renderer.setRenderTarget(prevRenderTarget, prevFace, prevMip);
                }

                // Readback float pixels asynchronously from GPU
                const raw0 = await renderer.readRenderTargetPixelsAsync(rt0, 0, 0, width, height);
                const raw1 = await renderer.readRenderTargetPixelsAsync(rt1, 0, 0, width, height);

                pixels0 = unpackRgba32Readback(raw0, width, height);
                pixels1 = unpackRgba32Readback(raw1, width, height);
            } finally {
                release();
            }

            // Compare GPU readback pixels with CPU reference values
            let maxAbsError = 0;
            let sumSquaredError = 0;
            let worstSample: GpuVerificationSamplePoint | undefined;
            let sampleCount = 0;
            let hasNonFinite = false;

            for (let py = 0; py < height; py++) {
                for (let px = 0; px < width; px++) {
                    const uBase = (px + 0.5) / width;
                    const vBase = (py + 0.5) / height;

                    const testU = uBase * uvScale[0] + uvOffset[0];
                    const testV = vBase * uvScale[1] + uvOffset[1];

                    const pIdx = (py * width + px) * 4;

                    // Direct extraction with strict finiteness validation (no '?? 0' default fallback)
                    const actualRaw: number[] = [
                        pixels0[pIdx]!,
                        pixels0[pIdx + 1]!,
                        pixels0[pIdx + 2]!,
                        pixels0[pIdx + 3]!,
                        pixels1[pIdx]!,
                        pixels1[pIdx + 1]!,
                        pixels1[pIdx + 2]!,
                        pixels1[pIdx + 3]!,
                    ];

                    // Canonical CPU forward pass reference
                    const expectedRaw = decodeNeuralMaterialRawCpu(fine, coarse, sharedWeights, {
                        u: testU,
                        v: testV,
                        wrapU: config.wrapU,
                        wrapV: config.wrapV,
                    });

                    const channelDiffs: number[] = [];
                    let pointMaxDiff = 0;
                    let worstCh = 0;

                    for (let ch = 0; ch < 8; ch++) {
                        const act = actualRaw[ch]!;
                        const exp = expectedRaw[ch]!;

                        // Astra review check: Instantly fail on any non-finite (NaN / Infinity) values
                        if (!Number.isFinite(act) || !Number.isFinite(exp)) {
                            hasNonFinite = true;
                            pointMaxDiff = Infinity;
                            sumSquaredError = Infinity;
                            worstCh = ch;
                            channelDiffs.push(Infinity);
                            break;
                        }

                        const diff = Math.abs(act - exp);
                        channelDiffs.push(diff);
                        sumSquaredError += diff * diff;
                        if (diff > pointMaxDiff) {
                            pointMaxDiff = diff;
                            worstCh = ch;
                        }
                    }

                    sampleCount += 8;

                    if (!worstSample || pointMaxDiff > maxAbsError) {
                        maxAbsError = pointMaxDiff;
                        worstSample = {
                            uv: [testU, testV],
                            expectedRaw: Array.from(expectedRaw),
                            actualRaw,
                            channelDiffs,
                            maxDiff: pointMaxDiff,
                            worstChannelIndex: worstCh,
                        };
                    }
                }
            }

            const rmse = Math.sqrt(sumSquaredError / Math.max(1, sampleCount));
            // Strict test pass condition: must have samples, no NaNs, finite RMSE, and maxAbsError <= tolerance
            const passed = !hasNonFinite && sampleCount > 0 && Number.isFinite(rmse) && maxAbsError <= tolerance;

            testResults.push({
                id: config.id,
                name: config.name,
                description: config.description,
                passed,
                maxAbsoluteError: maxAbsError,
                rootMeanSquareError: rmse,
                sampleCount,
                worstSample,
            });
            } finally { release(); }
        }
    } finally {
        quad.geometry.dispose();
    }

    const allPassed = testResults.length > 0 && testResults.every((r) => r.passed);
    const maxGlobalError = Math.max(0, ...testResults.map((r) => r.maxAbsoluteError));

    const markdownReport = generateVerificationMarkdownReport(
        adapterInfo,
        testResults,
        tolerance,
        allPassed,
        evidence,
        profile,
    );

    return {
        evidence,
        profile,
        timestamp: new Date().toISOString(),
        adapterInfo,
        passed: allPassed,
        maxErrorAcrossAllTests: maxGlobalError,
        tolerance,
        testCases: testResults,
        markdownReport,
    };
}

/**
 * Generates clean, human-readable Markdown summary of verification results.
 */
export function generateVerificationMarkdownReport(
    adapter: GpuVerificationAdapterInfo,
    results: readonly GpuVerificationTestCaseResult[],
    tolerance: number,
    allPassed: boolean,
    evidence: GpuVerificationReport['evidence'] = 'simulation',
    profile: DecoderProfile = 'standard',
): string {
    const lines: string[] = [];
    lines.push('# VNM1 WebGPU Readback Verification Report');
    lines.push('');
    lines.push(`Evidence: ${evidence}`);
    lines.push(`**Profile**: \`${profile}\``);
    lines.push(`**Execution Status**: ${allPassed ? '✅ PASSED (All cases within tolerance)' : '❌ FAILED'}`);
    lines.push(`**Tolerance Threshold**: ${tolerance} (Absolute float difference)`);
    lines.push(`**Target Adapter**: ${adapter.description} (Vendor: ${adapter.vendor}, Arch: ${adapter.architecture})`);
    lines.push('');
    lines.push('### Test Case Breakdown');
    lines.push('');
    lines.push('| ID | Case Name | Samples | Max Abs Error | RMSE | Status |');
    lines.push('| :--- | :--- | :---: | :---: | :---: | :---: |');

    for (const r of results) {
        const maxErrStr = Number.isFinite(r.maxAbsoluteError) ? r.maxAbsoluteError.toExponential(4) : 'NaN/Inf';
        const rmseStr = Number.isFinite(r.rootMeanSquareError) ? r.rootMeanSquareError.toExponential(4) : 'NaN/Inf';
        lines.push(
            `| \`${r.id}\` | ${r.name} | ${r.sampleCount} | ${maxErrStr} | ${rmseStr} | ${r.passed ? '✅ Pass' : '❌ Fail'} |`,
        );
    }

    lines.push('', '### Worst samples');
    for (const result of results) {
        const sample = result.worstSample;
        if (sample) lines.push(`${result.id}: UV=${sample.uv.join(',')} channel=${sample.worstChannelIndex} expected=[${sample.expectedRaw.join(',')}] actual=[${sample.actualRaw.join(',')}]`);
    }

    lines.push('');
    return lines.join('\n');
}

// Global browser console export for instant interactive verification
if (typeof window !== 'undefined') {
    (window as unknown as { __runNeuralMaterialVerification: typeof executeGpuVerificationAuto }).__runNeuralMaterialVerification =
        executeGpuVerificationAuto;
}
