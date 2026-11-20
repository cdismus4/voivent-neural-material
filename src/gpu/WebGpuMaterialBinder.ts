/**
 * WebGpuMaterialBinder.ts
 *
 * Portable Three.js WebGPU / TSL binder for Voivent Neural Material (VNM1).
 * Zero application-specific dependencies (safe for public standalone distribution).
 *
 * Lifecycle guarantees:
 * - Tier `tail`: Binds linear PBR DataTextures (baseColor, reconstructed normal, ORM)
 *   to MeshStandardNodeMaterial TSL slots for immediate low-bandwidth preview.
 * - Tier `level0`: Switches seamlessly to direct GPU fragment evaluation via
 *   `createNeuralMaterialTslNodes` + `attachNeuralPbrNodes` (no CPU baking).
 * - Re-binding or unbinding automatically disposes prior MaterialBindings, temporary
 *   DataTextures, and synthesized MeshStandardNodeMaterial instances while keeping
 *   the mesh's original pre-VNM material recoverable.
 */

import * as THREE from 'three';
import {
    float,
    normalMap,
    texture as textureNode,
    uv,
    vec4,
} from 'three/tsl';
import {
    ClampToEdgeWrapping,
    DataTexture,
    LinearFilter,
    MeshStandardNodeMaterial,
    NoColorSpace,
    RepeatWrapping,
    RGBAFormat,
    UnsignedByteType,
    type Texture,
} from 'three/webgpu';
import { resolveManifestMipMax } from '../format/types';
import type { ResidentNeuralAsset } from '../streaming/types';
import {
    attachNeuralPbrNodes,
    createDerivativeNormalizedMipNode,
    createLatentTextureSource,
    createNeuralMaterialTslNodes,
    createWeightsStorageBuffer,
    type LatentTextureSource,
    type MaterialBinding,
    type NeuralPbrNodes,
    type ScalarNode,
    type Vec2Node,
} from './TslMlpDecoderNode';

export type MipNormalizationMode = 'auto' | 'fixed-zero' | 'derivative';

export interface WebGpuBindOptions {
    /**
     * Controls how `normalizedMip` is supplied at runtime:
     * - `'auto'` (default): uses UV-derivative LOD calculation if `manifest.mipMax`
     *   is declared, otherwise supplies `0` for V1 assets trained at mip 0.
     * - `'fixed-zero'`: always supplies `0`.
     * - `'derivative'`: always computes `clamp(mip / mipMax, 0, 1)` in fragment shader.
     */
    readonly mipMode?: MipNormalizationMode;
    /** Explicit mipMax denominator override when `mipMode` uses derivatives. */
    readonly mipMax?: number;
    /** Explicit constant normalizedMip in [0, 1] (overrides `mipMode`). */
    readonly normalizedMip?: number;
}

export interface TailPreviewTextures {
    readonly baseColorTexture: DataTexture;
    readonly normalTexture: DataTexture;
    readonly ormTexture: DataTexture;
    readonly width: number;
    readonly height: number;
    dispose(): void;
}

export interface BoundMeshBatchResult {
    readonly tier: 'tail' | 'level0';
    readonly textureSize: { readonly width: number; readonly height: number };
    readonly boundMeshCount: number;
    dispose(): void;
}

interface ResourceBatch {
    retain(): void;
    release(): void;
}

interface MeshBindingRecord {
    readonly originalMaterial: THREE.Material | THREE.Material[];
    release(): void;
}

interface CopyableMaterialFields {
    name?: string;
    side?: THREE.Side;
    transparent?: boolean;
    opacity?: number;
    alphaTest?: number;
    depthWrite?: boolean;
    depthTest?: boolean;
    wireframe?: boolean;
    flatShading?: boolean;
    envMapIntensity?: number;
    vertexColors?: boolean;
}

const activeMeshBindings = new WeakMap<THREE.Mesh, MeshBindingRecord>();

function createRefCountedResourceBatch(onDispose: () => void): ResourceBatch {
    let refCount = 0;
    let disposed = false;
    return {
        retain() {
            if (disposed) {
                throw new Error('Cannot retain an already disposed VNM GPU resource batch');
            }
            refCount++;
        },
        release() {
            if (disposed) return;
            refCount--;
            if (refCount <= 0) {
                disposed = true;
                onDispose();
            }
        },
    };
}

function toTextureArray(source: LatentTextureSource): Texture[] {
    return Array.isArray(source) ? [...source] : [source as Texture];
}

function configurePreviewDataTexture(
    data: Uint8Array,
    width: number,
    height: number,
    wrapU: 'repeat' | 'clamp',
    wrapV: 'repeat' | 'clamp',
): DataTexture {
    const tex = new DataTexture(data, width, height, RGBAFormat, UnsignedByteType);
    tex.magFilter = LinearFilter;
    tex.minFilter = LinearFilter;
    tex.generateMipmaps = false;
    tex.colorSpace = NoColorSpace;
    tex.wrapS = wrapU === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;
    tex.wrapT = wrapV === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
}

/**
 * Builds linear PBR DataTextures (baseColor, tangent normal with Z reconstructed,
 * and packed ORM = [roughness, metalness, ao, 255]) directly from a `tail.bin` payload.
 */
export function createTailPreviewTextures(asset: ResidentNeuralAsset): TailPreviewTextures | null {
    if (!asset.tail) return null;

    const tail = asset.tail.tail;
    const width = tail.tailWidth;
    const height = tail.tailHeight;
    const texelCount = width * height;
    if (texelCount <= 0) return null;

    const src = tail.pixelData;
    const baseData = new Uint8Array(texelCount * 4);
    const normalData = new Uint8Array(texelCount * 4);
    const ormData = new Uint8Array(texelCount * 4);

    for (let i = 0; i < texelCount; i++) {
        const s = i * 8;
        const d = i * 4;

        baseData[d] = src[s] ?? 0;
        baseData[d + 1] = src[s + 1] ?? 0;
        baseData[d + 2] = src[s + 2] ?? 0;
        baseData[d + 3] = 255;

        // SPEC §5.3.4 / §6.4: tail stores encoded nx, ny in [0, 255]; reconstruct nz cleanly.
        let nx = (2 * (src[s + 3] ?? 128)) / 255 - 1;
        let ny = (2 * (src[s + 4] ?? 128)) / 255 - 1;
        nx = Math.max(-1, Math.min(1, nx));
        ny = Math.max(-1, Math.min(1, ny));
        const xySq = nx * nx + ny * ny;
        if (xySq > 1) {
            const inv = 1 / Math.sqrt(xySq);
            nx *= inv;
            ny *= inv;
        }
        const nz = Math.sqrt(Math.max(0, 1 - nx * nx - ny * ny));
        const normLen = Math.hypot(nx, ny, nz) || 1;
        normalData[d] = Math.round(((nx / normLen + 1) * 0.5) * 255);
        normalData[d + 1] = Math.round(((ny / normLen + 1) * 0.5) * 255);
        normalData[d + 2] = Math.round(((nz / normLen + 1) * 0.5) * 255);
        normalData[d + 3] = 255;

        ormData[d] = src[s + 5] ?? 128;
        ormData[d + 1] = src[s + 6] ?? 0;
        ormData[d + 2] = src[s + 7] ?? 255;
        ormData[d + 3] = 255;
    }

    const wrapU = asset.manifest?.wrapU ?? 'repeat';
    const wrapV = asset.manifest?.wrapV ?? 'repeat';

    let baseColorTexture: DataTexture | null = null;
    let normalTexture: DataTexture | null = null;
    let ormTexture: DataTexture | null = null;
    try {
        baseColorTexture = configurePreviewDataTexture(baseData, width, height, wrapU, wrapV);
        normalTexture = configurePreviewDataTexture(normalData, width, height, wrapU, wrapV);
        ormTexture = configurePreviewDataTexture(ormData, width, height, wrapU, wrapV);
    } catch (err) {
        baseColorTexture?.dispose();
        normalTexture?.dispose();
        ormTexture?.dispose();
        throw err;
    }

    const finalBase = baseColorTexture;
    const finalNormal = normalTexture;
    const finalOrm = ormTexture;

    let disposed = false;
    return {
        baseColorTexture: finalBase,
        normalTexture: finalNormal,
        ormTexture: finalOrm,
        width,
        height,
        dispose() {
            if (disposed) return;
            disposed = true;
            finalBase.dispose();
            finalNormal.dispose();
            finalOrm.dispose();
        },
    };
}

function createNodeMaterialFromSource(sourceMaterial: THREE.Material): MeshStandardNodeMaterial {
    const src = sourceMaterial as CopyableMaterialFields;
    const nodeMat = new MeshStandardNodeMaterial();

    if (typeof src.name === 'string' && src.name.length > 0) nodeMat.name = src.name;
    if (src.side !== undefined) nodeMat.side = src.side;
    if (typeof src.transparent === 'boolean') nodeMat.transparent = src.transparent;
    if (typeof src.opacity === 'number') nodeMat.opacity = src.opacity;
    if (typeof src.alphaTest === 'number') nodeMat.alphaTest = src.alphaTest;
    if (typeof src.depthWrite === 'boolean') nodeMat.depthWrite = src.depthWrite;
    if (typeof src.depthTest === 'boolean') nodeMat.depthTest = src.depthTest;
    if (typeof src.wireframe === 'boolean') nodeMat.wireframe = src.wireframe;
    if (typeof src.flatShading === 'boolean') nodeMat.flatShading = src.flatShading;
    if (typeof src.envMapIntensity === 'number') nodeMat.envMapIntensity = src.envMapIntensity;
    if (typeof src.vertexColors === 'boolean') nodeMat.vertexColors = src.vertexColors;

    // Neutral scalar factors so TSL PBR nodes express unattenuated decoded values.
    nodeMat.color.setHex(0xffffff);
    nodeMat.roughness = 1.0;
    nodeMat.metalness = 1.0;
    return nodeMat;
}

function resolveNormalizedMipNode(
    asset: ResidentNeuralAsset,
    uvNode: Vec2Node,
    options?: WebGpuBindOptions,
): ScalarNode {
    if (options?.normalizedMip !== undefined) {
        const constantMip = options.normalizedMip;
        if (!Number.isFinite(constantMip) || constantMip < 0 || constantMip > 1) {
            throw new Error(`normalizedMip must be within [0, 1], got '${String(constantMip)}'`);
        }
        return float(constantMip) as ScalarNode;
    }

    const mode: MipNormalizationMode = options?.mipMode ?? 'auto';
    const shouldUseDerivative =
        mode === 'derivative' ||
        (mode === 'auto' && asset.manifest.mipMax !== undefined);

    if (!shouldUseDerivative) {
        return float(0.0) as ScalarNode;
    }

    const mipMax = options?.mipMax ?? resolveManifestMipMax(asset.manifest);
    return createDerivativeNormalizedMipNode(uvNode, asset.manifest.sourceSize, mipMax);
}

function attachTailTexturesToNodeMaterial(
    nodeMaterial: MeshStandardNodeMaterial,
    tailTextures: TailPreviewTextures,
    uvNode: Vec2Node,
): MaterialBinding {
    const colorSample = textureNode(tailTextures.baseColorTexture, uvNode);
    const normalSample = textureNode(tailTextures.normalTexture, uvNode);
    const ormSample = textureNode(tailTextures.ormTexture, uvNode);

    const originalColorNode = nodeMaterial.colorNode;
    const originalNormalNode = nodeMaterial.normalNode;
    const originalRoughnessNode = nodeMaterial.roughnessNode;
    const originalMetalnessNode = nodeMaterial.metalnessNode;
    const originalAoNode = nodeMaterial.aoNode;

    let attached = true;
    nodeMaterial.colorNode = vec4(colorSample.rgb, float(1.0));
    nodeMaterial.normalNode = normalMap(vec4(normalSample.rgb, float(1.0)));
    nodeMaterial.roughnessNode = ormSample.x;
    nodeMaterial.metalnessNode = ormSample.y;
    nodeMaterial.aoNode = ormSample.z;
    nodeMaterial.needsUpdate = true;

    const restore = () => {
        if (!attached) return;
        attached = false;
        nodeMaterial.colorNode = originalColorNode;
        nodeMaterial.normalNode = originalNormalNode;
        nodeMaterial.roughnessNode = originalRoughnessNode;
        nodeMaterial.metalnessNode = originalMetalnessNode;
        nodeMaterial.aoNode = originalAoNode;
        nodeMaterial.needsUpdate = true;
    };

    return {
        restore,
        dispose: restore,
    };
}

/**
 * Binds a resident VNM1 asset directly to Three.js meshes using WebGPU / TSL nodes.
 * - `level0` uses `createNeuralMaterialTslNodes` + `attachNeuralPbrNodes` for GPU fragment decoding.
 * - `tail` uses `MeshStandardNodeMaterial` + linear `DataTexture` preview nodes.
 * Previous bindings and GPU textures on each mesh are automatically disposed upon re-bind.
 */
export function bindResidentAssetToMeshes(
    meshes: readonly THREE.Mesh[],
    asset: ResidentNeuralAsset,
    options?: WebGpuBindOptions,
): BoundMeshBatchResult | null {
    if (meshes.length === 0) return null;

    const uvNode = uv() as Vec2Node;
    let tier: 'tail' | 'level0';
    let textureSize: { width: number; height: number };
    let buildNodeBindingForSlot: (nodeMat: MeshStandardNodeMaterial) => MaterialBinding;
    let disposeSharedResources: () => void;

    if (asset.tier === 'level0' && asset.level0 && asset.weights) {
        const level0Meta = asset.manifest?.levels?.[0];
        if (!level0Meta) {
            throw new Error(`Manifest for '${asset.assetId}' is missing level[0] metadata required for GPU binding`);
        }

        const wrapU = asset.manifest.wrapU ?? 'repeat';
        const wrapV = asset.manifest.wrapV ?? 'repeat';
        const allocatedTextures: Texture[] = [];
        try {
            const fineSource = createLatentTextureSource(asset.level0.fine, asset.manifest.profile, wrapU, wrapV);
            allocatedTextures.push(...toTextureArray(fineSource));
            const coarseSource = createLatentTextureSource(asset.level0.coarse, asset.manifest.profile, wrapU, wrapV);
            allocatedTextures.push(...toTextureArray(coarseSource));
            const weightsBuffer = createWeightsStorageBuffer(asset.weights.weights);
            const normMipNode = resolveNormalizedMipNode(asset, uvNode, options);

            const nodes: NeuralPbrNodes = createNeuralMaterialTslNodes(
                fineSource,
                coarseSource,
                weightsBuffer,
                { uv: uvNode, normalizedMip: normMipNode },
                {
                    profile: asset.manifest.profile,
                    fineSize: level0Meta.fineSize,
                    coarseSize: level0Meta.coarseSize,
                    wrapU,
                    wrapV,
                },
            );

            tier = 'level0';
            textureSize = {
                width: asset.manifest.sourceSize[0],
                height: asset.manifest.sourceSize[1],
            };
            buildNodeBindingForSlot = (nodeMat) => attachNeuralPbrNodes(nodeMat, nodes);
            disposeSharedResources = () => {
                for (const tex of allocatedTextures) {
                    tex.dispose();
                }
            };
        } catch (err) {
            for (const tex of allocatedTextures) {
                tex.dispose();
            }
            throw err;
        }
    } else if (asset.tail) {
        const tailTextures = createTailPreviewTextures(asset);
        if (!tailTextures) return null;

        tier = 'tail';
        textureSize = { width: tailTextures.width, height: tailTextures.height };
        buildNodeBindingForSlot = (nodeMat) => attachTailTexturesToNodeMaterial(nodeMat, tailTextures, uvNode);
        disposeSharedResources = () => {
            tailTextures.dispose();
        };
    } else {
        return null;
    }

    let sharedDisposed = false;
    const disposeSharedOnce = () => {
        if (sharedDisposed) return;
        sharedDisposed = true;
        disposeSharedResources();
    };

    const batch = createRefCountedResourceBatch(disposeSharedOnce);
    const meshHandles: Array<() => void> = [];

    try {
        for (const mesh of meshes) {
            const previousRecord = activeMeshBindings.get(mesh);
            const originalMaterial = previousRecord ? previousRecord.originalMaterial : mesh.material;

            const slotCleanup: Array<() => void> = [];
            const convertSlot = (srcMat: THREE.Material): MeshStandardNodeMaterial => {
                const nodeMat = createNodeMaterialFromSource(srcMat);
                try {
                    const binding = buildNodeBindingForSlot(nodeMat);
                    slotCleanup.push(() => {
                        binding.dispose();
                        nodeMat.dispose();
                    });
                    return nodeMat;
                } catch (err) {
                    nodeMat.dispose();
                    throw err;
                }
            };

            let nextMaterial: THREE.Material | THREE.Material[];
            try {
                if (Array.isArray(originalMaterial)) {
                    nextMaterial = originalMaterial.map((slot) => (slot ? convertSlot(slot) : slot));
                } else if (originalMaterial) {
                    nextMaterial = convertSlot(originalMaterial);
                } else {
                    nextMaterial = originalMaterial;
                }
            } catch (err) {
                for (const cleanup of slotCleanup) {
                    cleanup();
                }
                throw err;
            }

            // Dispose any prior VNM binding and GPU textures attached to this mesh only after new slots succeed.
            if (previousRecord) {
                previousRecord.release();
                activeMeshBindings.delete(mesh);
            }

            mesh.material = nextMaterial;
            batch.retain();

            let released = false;
            const releaseMeshBinding = () => {
                if (released) return;
                released = true;
                for (const cleanup of slotCleanup) {
                    cleanup();
                }
                batch.release();
            };

            const record: MeshBindingRecord = {
                originalMaterial,
                release: releaseMeshBinding,
            };
            activeMeshBindings.set(mesh, record);

            meshHandles.push(() => {
                if (activeMeshBindings.get(mesh) === record) {
                    mesh.material = originalMaterial;
                    activeMeshBindings.delete(mesh);
                }
                releaseMeshBinding();
            });
        }
    } catch (err) {
        for (const releaseHandle of meshHandles) {
            releaseHandle();
        }
        disposeSharedOnce();
        throw err;
    }

    return {
        tier,
        textureSize,
        boundMeshCount: meshes.length,
        dispose() {
            for (const releaseHandle of meshHandles) {
                releaseHandle();
            }
        },
    };
}

/**
 * Unbinds VNM1 materials from meshes, restores their pre-VNM materials, and disposes GPU resources.
 */
export function unbindResidentAssetFromMeshes(meshes: readonly THREE.Mesh[]): number {
    let unboundCount = 0;
    for (const mesh of meshes) {
        const record = activeMeshBindings.get(mesh);
        if (!record) continue;
        mesh.material = record.originalMaterial;
        activeMeshBindings.delete(mesh);
        record.release();
        unboundCount++;
    }
    return unboundCount;
}
