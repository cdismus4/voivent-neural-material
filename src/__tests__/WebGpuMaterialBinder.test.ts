import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { uv } from 'three/tsl';
import {
    bindResidentAssetToMeshes,
    createDerivativeNormalizedMipNode,
    createTailPreviewTextures,
    MLP_PROFILES,
    parseManifest,
    resolveManifestMipMax,
    unbindResidentAssetFromMeshes,
    type DecoderProfile,
    type MlpWeights,
    type NeuralMaterialManifest,
    type ResidentNeuralAsset,
    type Vec2Node,
} from '../index';

function makeManifest(profile: DecoderProfile, overrides?: Partial<NeuralMaterialManifest>): NeuralMaterialManifest {
    const channels = (MLP_PROFILES[profile].inputDim - 4) / 2;
    return {
        format: 'VNM1',
        version: 1,
        assetId: `pkg_${profile}_asset`,
        sourceSize: [512, 512],
        profile,
        quantization: 'u8',
        weightStoragePrecision: 'f32',
        computePrecision: 'f32',
        channels: 'base3-normal2-rough-metal-ao',
        baseColorSpace: 'linear-srgb',
        normalConvention: 'tangent-positive-y-positive-z',
        wrapU: 'repeat',
        wrapV: 'repeat',
        weights: { url: 'weights.bin', sha256: '0'.repeat(64), byteLength: MLP_PROFILES[profile].totalWeights * 4 + 16 },
        levels: [
            {
                sourceMip: 0,
                sourceSize: [512, 512],
                fineSize: [32, 32],
                coarseSize: [16, 16],
                chunk: { url: 'level-0.bin', sha256: '1'.repeat(64), byteLength: (32 * 32 + 16 * 16) * channels + 32 },
            },
        ],
        tailFirstMip: 5,
        tail: { url: 'tail.bin', sha256: '2'.repeat(64), byteLength: 16 * 16 * 8 + 32 },
        ...overrides,
    };
}

function makeWeights(profile: DecoderProfile): MlpWeights {
    const spec = MLP_PROFILES[profile];
    return {
        profile,
        w1: new Float32Array(spec.hidden1 * spec.inputDim).fill(0.01),
        b1: new Float32Array(spec.hidden1).fill(0.02),
        w2: new Float32Array(spec.hidden2 * spec.hidden1).fill(0.01),
        b2: new Float32Array(spec.hidden2).fill(0.02),
        w3: new Float32Array(spec.outputDim * spec.hidden2).fill(0.01),
        b3: new Float32Array(spec.outputDim).fill(0.5),
    };
}

function makeTailAsset(assetId: string): ResidentNeuralAsset {
    const pixelData = new Uint8Array(16 * 16 * 8).fill(180);
    return {
        assetId,
        manifest: makeManifest('standard', { assetId }),
        tier: 'tail',
        state: 'resident',
        tail: {
            tail: {
                header: { magic: 0x314d4e56, version: 1, chunkKind: 3, payloadByteLength: 16 + pixelData.byteLength },
                tailWidth: 16,
                tailHeight: 16,
                tailFirstMip: 5,
                channels: 8,
                format: 1,
                pixelData,
            },
            rawBytes: new Uint8Array(100),
        },
        cpuByteSize: 1000,
        gpuByteSize: 2048,
        referenceCount: 1,
        generation: 1,
        lastAccessedAt: Date.now(),
    };
}

function makeLevel0Asset(profile: DecoderProfile, overrides?: Partial<NeuralMaterialManifest>): ResidentNeuralAsset {
    const manifest = makeManifest(profile, overrides);
    const channels = (MLP_PROFILES[profile].inputDim - 4) / 2;
    const fineData = new Uint8Array(32 * 32 * channels).fill(128);
    const coarseData = new Uint8Array(16 * 16 * channels).fill(128);
    return {
        assetId: manifest.assetId,
        manifest,
        tier: 'level0',
        state: 'resident',
        weights: { weights: makeWeights(profile), rawBytes: new Uint8Array(64) },
        level0: {
            fine: { width: 32, height: 32, channels, data: fineData, isNormalizedU8: true },
            coarse: { width: 16, height: 16, channels, data: coarseData, isNormalizedU8: true },
            rawBytes: new Uint8Array(128),
        },
        cpuByteSize: fineData.byteLength + coarseData.byteLength,
        gpuByteSize: fineData.byteLength + coarseData.byteLength,
        referenceCount: 1,
        generation: 2,
        lastAccessedAt: Date.now(),
    };
}

describe('WebGpuMaterialBinder Standalone Suite', () => {
    it('promotes tail preview to level0 TSL shader and cleanly disposes/restores on unbind', () => {
        const origMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), origMat);

        const tailAsset = makeTailAsset('standalone_tail');
        const preview = createTailPreviewTextures(tailAsset);
        expect(preview).not.toBeNull();
        preview!.dispose();

        const tailHandle = bindResidentAssetToMeshes([mesh], tailAsset);
        expect(tailHandle?.tier).toBe('tail');
        const tailMatUnknown: unknown = mesh.material;
        expect(tailMatUnknown).toBeInstanceOf(MeshStandardNodeMaterial);
        if (!(tailMatUnknown instanceof MeshStandardNodeMaterial)) {
            throw new Error('Expected MeshStandardNodeMaterial');
        }
        const tailDisposeSpy = vi.spyOn(tailMatUnknown, 'dispose');

        const level0Handle = bindResidentAssetToMeshes([mesh], makeLevel0Asset('quality8'));
        expect(level0Handle?.tier).toBe('level0');
        expect(tailDisposeSpy).toHaveBeenCalledTimes(1);

        // F-1 regression: disposing an older batch handle must NOT detach or leak the active newer binding
        const level0MatUnknown: unknown = mesh.material;
        expect(level0MatUnknown).toBeInstanceOf(MeshStandardNodeMaterial);
        const level0DisposeSpy = vi.spyOn(level0MatUnknown as MeshStandardNodeMaterial, 'dispose');
        tailHandle!.dispose();
        expect(mesh.material).toBe(level0MatUnknown);
        expect(level0DisposeSpy).not.toHaveBeenCalled();

        expect(unbindResidentAssetFromMeshes([mesh])).toBe(1);
        expect(level0DisposeSpy).toHaveBeenCalledTimes(1);
        expect(mesh.material).toBe(origMat);
    });

    it('F-1/M-1: cleans up allocated GPU textures and partial bindings when binding throws midway', () => {
        const origMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
        const mesh1 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), origMat);
        const poisonMat = new THREE.MeshStandardMaterial();
        Object.defineProperty(poisonMat, 'name', {
            get() {
                throw new Error('simulated slot conversion error');
            },
        });
        const mesh2 = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), poisonMat);

        const texDisposeSpy = vi.spyOn(THREE.DataTexture.prototype, 'dispose');
        try {
            // Throw after fine/coarse latent textures are allocated (invalid normalizedMip)
            expect(() => bindResidentAssetToMeshes([mesh1], makeLevel0Asset('quality8'), { normalizedMip: -1 })).toThrow(/normalizedMip/);
            expect(texDisposeSpy).toHaveBeenCalledTimes(4); // quality8 allocates 2 fine + 2 coarse DataTextures
            expect(mesh1.material).toBe(origMat);

            texDisposeSpy.mockClear();
            // Throw mid-mesh-loop on mesh2: mesh1 and all shared DataTextures must roll back cleanly
            expect(() => bindResidentAssetToMeshes([mesh1, mesh2], makeLevel0Asset('standard'))).toThrow(/simulated slot conversion error/);
            expect(texDisposeSpy).toHaveBeenCalledTimes(2); // standard allocates 1 fine + 1 coarse DataTexture
            expect(mesh1.material).toBe(origMat);
            expect(unbindResidentAssetFromMeshes([mesh1, mesh2])).toBe(0);
        } finally {
            texDisposeSpy.mockRestore();
        }
    });

    it('supports Option C mipMax resolution and UV-derivative normalizedMip TSL node', () => {
        expect(resolveManifestMipMax({ tailFirstMip: 7 })).toBe(6);
        expect(resolveManifestMipMax({ tailFirstMip: 7, mipMax: 9 })).toBe(9);
        const parsed = parseManifest(makeManifest('standard8', { mipMax: 7 }));
        expect(parsed.mipMax).toBe(7);
        expect(createDerivativeNormalizedMipNode(uv() as Vec2Node, [1024, 1024], 7)).toBeTruthy();
    });
});
