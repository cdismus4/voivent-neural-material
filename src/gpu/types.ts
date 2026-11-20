import type { DecoderProfile } from '../format/types';

/**
 * Types and interfaces for real GPU WebGPU readback verification harness (Ticket B2 / P2).
 */

export interface GpuVerificationSamplePoint {
    readonly uv: readonly [number, number];
    readonly expectedRaw: readonly number[]; // 8 floats
    readonly actualRaw: readonly number[];   // 8 floats
    readonly channelDiffs: readonly number[]; // 8 absolute diffs
    readonly maxDiff: number;
    readonly worstChannelIndex: number;
}

export interface GpuVerificationTestCaseResult {
    readonly id: string;
    readonly name: string;
    readonly description: string;
    readonly passed: boolean;
    readonly maxAbsoluteError: number;
    readonly rootMeanSquareError: number;
    readonly sampleCount: number;
    readonly worstSample?: GpuVerificationSamplePoint;
}

export interface GpuVerificationAdapterInfo {
    readonly vendor: string;
    readonly architecture: string;
    readonly device: string;
    readonly description: string;
}

export interface GpuVerificationReport {
    readonly evidence: 'simulation' | 'webgpu-readback';
    readonly timestamp: string;
    readonly profile?: DecoderProfile;
    readonly adapterInfo: GpuVerificationAdapterInfo;
    readonly passed: boolean;
    readonly maxErrorAcrossAllTests: number;
    readonly tolerance: number;
    readonly testCases: readonly GpuVerificationTestCaseResult[];
    readonly markdownReport: string;
}

export interface GpuVerificationOptions {
    /** Target VNM1 profile to verify. Defaults to 'standard' for backwards compatibility. */
    readonly profile?: DecoderProfile;
    /** Maximum acceptable absolute difference between CPU reference and GPU readback. Defaults to 1e-3. */
    readonly tolerance?: number;
    /** Grid resolution, integer 1..256 (e.g. 16 for 256 pixels). */
    readonly resolution?: number;
    /** Optional test case filters. If omitted, all test cases run. */
    readonly filterCaseIds?: readonly string[];
}
