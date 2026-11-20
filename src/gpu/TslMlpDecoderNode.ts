import {
    clamp,
    dFdx,
    dFdy,
    dot,
    float,
    fract,
    length,
    log2,
    max,
    normalize,
    normalMap,
    sqrt,
    storage,
    texture as textureNode,
    vec2,
    vec3,
    vec4,
} from 'three/tsl';
import {
    ClampToEdgeWrapping,
    DataTexture,
    LinearFilter,
    NoColorSpace,
    RepeatWrapping,
    RGBAFormat,
    StorageBufferAttribute,
    UnsignedByteType,
    type MeshStandardNodeMaterial,
    type Node,
    type Texture,
} from 'three/webgpu';
import type { DecoderProfile, LatentFeatureGrid, MlpWeights } from '../format/types';
import { getLatentChannelCount, MLP_PROFILES } from '../format/types';
import { validateMlpWeights, validateU8LatentGrid } from '../format/validation';

export type ScalarNode = Node<'float'>;
export type Vec2Node = Node<'vec2'>;
export type Vec3Node = Node<'vec3'>;
export type Vec4Node = Node<'vec4'>;

export interface NeuralSampleContext {
    readonly uv: Vec2Node;
    readonly uvDx?: Vec2Node;
    readonly uvDy?: Vec2Node;
    readonly normalizedMip?: ScalarNode;
}

export interface NeuralPbrNodes {
    /** Raw 8 un-clamped outputs from MLP [y0..y3, y4..y7] for precision and loss evaluation */
    readonly rawOutputs: readonly [Vec4Node, Vec4Node];
    readonly baseColorLinear: Vec3Node;
    readonly normalTangent: Vec3Node;
    readonly roughness: ScalarNode;
    readonly metalness: ScalarNode;
    readonly ao: ScalarNode;
}

export interface MaterialBinding {
    restore(): void;
    dispose(): void;
}

export type LatentTextureSource = Texture | readonly [Texture, Texture];

export interface TslDecoderOptions {
    /** Target VNM1 profile; inferred from `weights.profile` when MlpWeights is passed, defaults to `'standard'` for raw StorageBufferAttribute. */
    readonly profile?: DecoderProfile;
    readonly fineSize: readonly [number, number];
    readonly coarseSize?: readonly [number, number];
    readonly normalizedMip?: number;
    readonly wrapU?: 'repeat' | 'clamp';
    readonly wrapV?: 'repeat' | 'clamp';
}

function buildRgba8DataTexture(
    buffer: Uint8Array,
    width: number,
    height: number,
    wrapU: 'repeat' | 'clamp',
    wrapV: 'repeat' | 'clamp',
): DataTexture {
    const texture = new DataTexture(
        buffer,
        width,
        height,
        RGBAFormat,
        UnsignedByteType,
    );
    texture.magFilter = LinearFilter;
    texture.minFilter = LinearFilter;
    texture.generateMipmaps = false;
    texture.colorSpace = NoColorSpace;
    texture.wrapS = wrapU === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;
    texture.wrapT = wrapV === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;
    texture.needsUpdate = true;
    return texture;
}

/**
 * Creates an optimized RGBA8 DataTexture for 4-channel latent feature grids.
 * Enforces LinearFilter, NoColorSpace, no auto-mipmaps, and explicit wrap modes.
 * Strictly requires normalized u8 encoding in VNM1.
 */
export function createLatentDataTexture(
    grid: LatentFeatureGrid,
    wrapU: 'repeat' | 'clamp' = 'repeat',
    wrapV: 'repeat' | 'clamp' = 'repeat',
): DataTexture {
    validateU8LatentGrid(grid, 4);
    if (!grid.isNormalizedU8 || !(grid.data instanceof Uint8Array)) {
        throw new Error('createLatentDataTexture strictly requires normalized u8 LatentFeatureGrid (Uint8Array)');
    }
    const expectedBytes = grid.width * grid.height * 4;
    if (grid.data.byteLength !== expectedBytes) {
        throw new Error(`Latent grid byteLength mismatch: expected ${expectedBytes}, got ${grid.data.byteLength}`);
    }

    // Allocate dedicated Uint8Array buffer to prevent subview retention
    const buffer = new Uint8Array(expectedBytes);
    buffer.set(grid.data);
    return buildRgba8DataTexture(buffer, grid.width, grid.height, wrapU, wrapV);
}

/**
 * Splits an 8-channel latent feature grid (standard8 / quality8) into a pair of RGBA8 DataTextures
 * ([channels 0..3, channels 4..7]) with matching LinearFilter and wrap alignment.
 */
export function createLatentDataTexturePair(
    grid: LatentFeatureGrid,
    wrapU: 'repeat' | 'clamp' = 'repeat',
    wrapV: 'repeat' | 'clamp' = 'repeat',
): readonly [DataTexture, DataTexture] {
    validateU8LatentGrid(grid, 8);
    if (!grid.isNormalizedU8 || !(grid.data instanceof Uint8Array)) {
        throw new Error('createLatentDataTexturePair strictly requires normalized u8 LatentFeatureGrid (Uint8Array)');
    }
    const texelCount = grid.width * grid.height;
    const buf0 = new Uint8Array(texelCount * 4);
    const buf1 = new Uint8Array(texelCount * 4);
    const src = grid.data;
    for (let i = 0; i < texelCount; i++) {
        const s = i * 8;
        const d = i * 4;
        buf0[d + 0] = src[s + 0]!;
        buf0[d + 1] = src[s + 1]!;
        buf0[d + 2] = src[s + 2]!;
        buf0[d + 3] = src[s + 3]!;
        buf1[d + 0] = src[s + 4]!;
        buf1[d + 1] = src[s + 5]!;
        buf1[d + 2] = src[s + 6]!;
        buf1[d + 3] = src[s + 7]!;
    }
    return [
        buildRgba8DataTexture(buf0, grid.width, grid.height, wrapU, wrapV),
        buildRgba8DataTexture(buf1, grid.width, grid.height, wrapU, wrapV),
    ];
}

/**
 * Creates the profile-appropriate GPU texture input (1 RGBA8 texture for 4ch profiles,
 * 2 RGBA8 textures for 8ch profiles).
 */
export function createLatentTextureSource(
    grid: LatentFeatureGrid,
    profile: DecoderProfile,
    wrapU: 'repeat' | 'clamp' = 'repeat',
    wrapV: 'repeat' | 'clamp' = 'repeat',
): LatentTextureSource {
    const ch = getLatentChannelCount(profile);
    if (ch === 4) return createLatentDataTexture(grid, wrapU, wrapV);
    if (ch === 8) return createLatentDataTexturePair(grid, wrapU, wrapV);
    throw new Error(`Unsupported GPU latent channel count: ${ch} for profile '${profile}'`);
}

/**
 * Packs MLP weights into a read-only WebGPU StorageBufferAttribute (vec4 array).
 * Ensures identical shader code across materials with zero shader string constant recompilation.
 */
export function createWeightsStorageBuffer(weights: MlpWeights): StorageBufferAttribute {
    const spec = validateMlpWeights(weights);
    if (spec.totalWeights % 4 !== 0) {
        throw new Error(`Profile '${weights.profile}' totalWeights (${spec.totalWeights}) must be a multiple of 4 for vec4 packing`);
    }
    const buf = new Float32Array(spec.totalWeights);
    let offset = 0;

    buf.set(weights.w1, offset); offset += weights.w1.length;
    buf.set(weights.b1, offset); offset += weights.b1.length;
    buf.set(weights.w2, offset); offset += weights.w2.length;
    buf.set(weights.b2, offset); offset += weights.b2.length;
    buf.set(weights.w3, offset); offset += weights.w3.length;
    buf.set(weights.b3, offset);

    return new StorageBufferAttribute(buf, 4);
}

const VEC4_COMPONENTS = ['x', 'y', 'z', 'w'] as const;

function packScalarNodesToVec4(nodes: readonly ScalarNode[]): Vec4Node[] {
    if (nodes.length % 4 !== 0) {
        throw new Error(`Expected node count divisible by 4, got ${nodes.length}`);
    }
    const packed: Vec4Node[] = [];
    for (let k = 0; k < nodes.length; k += 4) {
        packed.push(vec4(nodes[k]!, nodes[k + 1]!, nodes[k + 2]!, nodes[k + 3]!) as Vec4Node);
    }
    return packed;
}

function evalMlpLayerVec4(
    weightBuffer: ReturnType<typeof storage<'vec4'>>,
    weightStartVec4: number,
    biasStartVec4: number,
    outDim: number,
    inputVecs: readonly Vec4Node[],
    applyRelu: boolean,
): ScalarNode[] {
    const inVec4Count = inputVecs.length;
    const out: ScalarNode[] = [];
    for (let i = 0; i < outDim; i++) {
        const biasVec = weightBuffer.element(biasStartVec4 + Math.floor(i / 4));
        const biasComp = VEC4_COMPONENTS[i % 4]!;
        let sum: ScalarNode = biasVec[biasComp];
        const rowBase = weightStartVec4 + i * inVec4Count;
        for (let k = 0; k < inVec4Count; k++) {
            const wk = weightBuffer.element(rowBase + k);
            sum = sum.add(dot(wk, inputVecs[k]!)) as ScalarNode;
        }
        out.push(applyRelu ? (max(float(0.0), sum) as ScalarNode) : sum);
    }
    return out;
}

function normalizeLatentTextureList(source: LatentTextureSource): readonly Texture[] {
    return Array.isArray(source) ? (source as readonly Texture[]) : [source as Texture];
}

function validateLatentTextureEntry(
    texture: Texture,
    expectedSize: readonly [number, number] | undefined,
    expectedWrapS: typeof RepeatWrapping | typeof ClampToEdgeWrapping,
    expectedWrapT: typeof RepeatWrapping | typeof ClampToEdgeWrapping,
    label: string,
): void {
    if (!(texture instanceof DataTexture) || texture.type !== UnsignedByteType || texture.format !== RGBAFormat ||
        texture.minFilter !== LinearFilter || texture.magFilter !== LinearFilter || texture.colorSpace !== NoColorSpace ||
        texture.generateMipmaps || texture.mipmaps.length > 0 || texture.flipY) {
        throw new Error(`${label} must use unflipped linear-filtered RGBA u8 data without mipmaps`);
    }
    const { data, width, height } = texture.image;
    if (!(data instanceof Uint8Array)) throw new Error(`${label} data must be Uint8Array`);
    validateU8LatentGrid({ data, width, height, channels: 4, isNormalizedU8: true }, 4);
    if (expectedSize && (expectedSize[0] !== width || expectedSize[1] !== height)) {
        throw new Error(`${label} dimensions (${width}x${height}) do not match decoder options (${expectedSize[0]}x${expectedSize[1]})`);
    }
    if (texture.wrapS !== expectedWrapS || texture.wrapT !== expectedWrapT) {
        throw new Error(`${label} wrap mismatch: texture has wrapS=${texture.wrapS}, wrapT=${texture.wrapT}, expected ${expectedWrapS}, ${expectedWrapT}`);
    }
}

/**
 * Computes hardware UV-derivative normalizedMip in [0, 1] per SPEC §6.2:
 * rho = max(length(dFdx(uv) * sourceSize), length(dFdy(uv) * sourceSize)),
 * mip = log2(max(rho, 1)), normalizedMip = clamp(mip / mipMax, 0, 1).
 */
export function createDerivativeNormalizedMipNode(
    uvNode: Vec2Node,
    sourceSize: readonly [number, number],
    mipMax: number,
): ScalarNode {
    if (!Number.isFinite(mipMax) || mipMax <= 0) {
        throw new Error(`mipMax must be a positive finite number, got '${String(mipMax)}'`);
    }
    if (!Number.isFinite(sourceSize[0]) || sourceSize[0] <= 0 || !Number.isFinite(sourceSize[1]) || sourceSize[1] <= 0) {
        throw new Error('sourceSize dimensions must be positive finite numbers');
    }
    const scale = vec2(sourceSize[0], sourceSize[1]);
    const dx = length(dFdx(uvNode).mul(scale));
    const dy = length(dFdy(uvNode).mul(scale));
    const rho = max(dx, dy);
    const mip = log2(max(rho, float(1.0)));
    return clamp(mip.div(float(mipMax)), float(0.0), float(1.0)) as ScalarNode;
}

/**
 * Creates pure TSL Node graph implementing the VNM1 MLP decoder generalized across all
 * registered 4ch and 8ch profiles (small, standard, quality, standard8, quality8).
 * Reads weights from a WebGPU storage buffer node, eliminating hardcoded shader constants.
 */
export function createNeuralMaterialTslNodes(
    fineTexture: LatentTextureSource,
    coarseTexture: LatentTextureSource,
    weights: MlpWeights | StorageBufferAttribute,
    context: NeuralSampleContext,
    options: TslDecoderOptions,
): NeuralPbrNodes {
    let profile: DecoderProfile;
    if ('profile' in weights) {
        profile = weights.profile;
        if (options.profile !== undefined && options.profile !== profile) {
            throw new Error(`Profile mismatch: options.profile='${options.profile}' does not match weights.profile='${profile}'`);
        }
    } else {
        profile = options.profile ?? 'standard';
    }

    const spec = MLP_PROFILES[profile];
    if (!spec) {
        throw new Error(`Unsupported profile in GPU decoder: '${String(profile)}'`);
    }
    if (spec.inputDim % 4 !== 0 || spec.hidden1 % 4 !== 0 || spec.hidden2 % 4 !== 0 || spec.outputDim !== 8 || spec.totalWeights % 4 !== 0) {
        throw new Error(`Profile '${profile}' dimensions are incompatible with vec4 GPU layout`);
    }

    const expectedVec4s = spec.totalWeights / 4;
    if (!('profile' in weights)) {
        if (weights.count !== expectedVec4s || weights.itemSize !== 4) {
            throw new Error(`StorageBufferAttribute must have exactly ${expectedVec4s} vec4 items for profile '${profile}' (got count=${weights.count}, itemSize=${weights.itemSize})`);
        }
        if (!(weights.array instanceof Float32Array) || !weights.array.every(Number.isFinite) || weights.normalized) {
            throw new Error('Weight storage must contain finite, non-normalized f32 values');
        }
    }

    const channelsPerGrid = getLatentChannelCount(profile);
    if (channelsPerGrid !== 4 && channelsPerGrid !== 8) {
        throw new Error(`Unsupported GPU latent channel count ${channelsPerGrid} for profile '${profile}'`);
    }
    const expectedTexCount = channelsPerGrid / 4;
    const fineTextures = normalizeLatentTextureList(fineTexture);
    const coarseTextures = normalizeLatentTextureList(coarseTexture);
    if (fineTextures.length !== expectedTexCount || coarseTextures.length !== expectedTexCount) {
        throw new Error(
            `Profile '${profile}' (${channelsPerGrid}ch) requires ${expectedTexCount} RGBA8 texture(s) per latent grid (got fine=${fineTextures.length}, coarse=${coarseTextures.length})`,
        );
    }

    const wrapU = options.wrapU ?? 'repeat';
    const wrapV = options.wrapV ?? 'repeat';
    const expectedWrapS = wrapU === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;
    const expectedWrapT = wrapV === 'clamp' ? ClampToEdgeWrapping : RepeatWrapping;

    fineTextures.forEach((tex, idx) => validateLatentTextureEntry(tex, options.fineSize, expectedWrapS, expectedWrapT, `fineTexture[${idx}]`));
    coarseTextures.forEach((tex, idx) => validateLatentTextureEntry(tex, options.coarseSize, expectedWrapS, expectedWrapT, `coarseTexture[${idx}]`));

    const mip = options.normalizedMip ?? 0;
    if (!Number.isFinite(mip) || mip < 0 || mip > 1) throw new Error('normalizedMip must be within [0, 1]');

    const storageAttr = 'profile' in weights ? createWeightsStorageBuffer(weights) : weights;
    const weightBuffer = storage(storageAttr, 'vec4', expectedVec4s).toReadOnly();

    const uvNode = context.uv;
    const [fineW, fineH] = options.fineSize;
    const normMip = context.normalizedMip ?? float(mip);

    // 1. Sample Fine & Coarse latent textures at LOD 0, then dequantize: f = 2 * sample - 1
    const fineFeats = fineTextures.map((tex) => textureNode(tex, uvNode, float(0)).mul(2.0).sub(1.0) as Vec4Node);
    const coarseFeats = coarseTextures.map((tex) => textureNode(tex, uvNode, float(0)).mul(2.0).sub(1.0) as Vec4Node);

    // 2. Local coordinates inside fine grid texel with explicit wrap alignment:
    //    p = uvEffective * fineSize - 0.5; r = 2 * fract(p) - 1
    const uWrapped = wrapU === 'repeat' ? fract(uvNode.x) : clamp(uvNode.x, float(0.0), float(1.0));
    const vWrapped = wrapV === 'repeat' ? fract(uvNode.y) : clamp(uvNode.y, float(0.0), float(1.0));
    const uvEffective = vec2(uWrapped, vWrapped);

    const p = uvEffective.mul(vec2(fineW, fineH)).sub(0.5);
    const r = fract(p).mul(2.0).sub(1.0);

    // 3. Pack input vec4s:
    //    4ch: [Fine0, Coarse0, (rx, ry, normMip, 1)] (3 vec4s = 12 floats)
    //    8ch: [Fine0, Fine1, Coarse0, Coarse1, (rx, ry, normMip, 1)] (5 vec4s = 20 floats)
    const inputVecs: Vec4Node[] = [
        ...fineFeats,
        ...coarseFeats,
        vec4(r.x, r.y, normMip, float(1.0)) as Vec4Node,
    ];

    const inputVec4s = spec.inputDim / 4;
    const h1Vec4s = spec.hidden1 / 4;
    const h2Vec4s = spec.hidden2 / 4;

    const w1Start = 0;
    const b1Start = w1Start + spec.hidden1 * inputVec4s;
    const w2Start = b1Start + h1Vec4s;
    const b2Start = w2Start + spec.hidden2 * h1Vec4s;
    const w3Start = b2Start + h2Vec4s;
    const b3Start = w3Start + spec.outputDim * h2Vec4s;

    // Layer 1 (ReLU) -> pack -> Layer 2 (ReLU) -> pack -> Layer 3 (Linear)
    const h1Nodes = evalMlpLayerVec4(weightBuffer, w1Start, b1Start, spec.hidden1, inputVecs, true);
    const u = packScalarNodesToVec4(h1Nodes);
    const h2Nodes = evalMlpLayerVec4(weightBuffer, w2Start, b2Start, spec.hidden2, u, true);
    const q = packScalarNodesToVec4(h2Nodes);
    const outputNodes = evalMlpLayerVec4(weightBuffer, w3Start, b3Start, spec.outputDim, q, false);

    // Pack raw 8 un-clamped outputs into two vec4s
    const raw0 = vec4(outputNodes[0]!, outputNodes[1]!, outputNodes[2]!, outputNodes[3]!) as Vec4Node;
    const raw1 = vec4(outputNodes[4]!, outputNodes[5]!, outputNodes[6]!, outputNodes[7]!) as Vec4Node;

    // 4. PBR channels derivation matching CpuReferenceDecoder
    const baseColorLinear = clamp(
        vec3(outputNodes[0]!, outputNodes[1]!, outputNodes[2]!),
        float(0.0),
        float(1.0),
    ) as Vec3Node;

    const nxRaw = clamp(outputNodes[3]!, float(-1.0), float(1.0));
    const nyRaw = clamp(outputNodes[4]!, float(-1.0), float(1.0));
    const xySq = nxRaw.mul(nxRaw).add(nyRaw.mul(nyRaw));
    const invScale = float(1.0).div(sqrt(max(float(1.0), xySq)));
    const nx = nxRaw.mul(invScale);
    const ny = nyRaw.mul(invScale);
    const nz = sqrt(max(float(0.0), float(1.0).sub(xySq)));
    const normalTangent = normalize(vec3(nx, ny, nz)) as Vec3Node;

    const roughness = clamp(outputNodes[5]!, float(0.0), float(1.0)) as ScalarNode;
    const metalness = clamp(outputNodes[6]!, float(0.0), float(1.0)) as ScalarNode;
    const ao = clamp(outputNodes[7]!, float(0.0), float(1.0)) as ScalarNode;

    return {
        rawOutputs: [raw0, raw1],
        baseColorLinear,
        normalTangent,
        roughness,
        metalness,
        ao,
    };
}

/**
 * Attaches decoded neural PBR nodes to a Three.js MeshStandardNodeMaterial.
 * Returns an idempotent MaterialBinding to restore original bindings safely.
 */
export function attachNeuralPbrNodes(
    material: MeshStandardNodeMaterial,
    nodes: NeuralPbrNodes,
): MaterialBinding {
    // Astra review check: Enforce single-owner binding per material.
    // Prevent overriding another active neural binding and corrupting restoration stack.
    if (material.userData.__activeNeuralBindingToken) {
        throw new Error('Material already has an active neural material binding. Restore or dispose previous binding before attaching a new one.');
    }

    const currentToken = Symbol('neural-binding-token');
    material.userData.__activeNeuralBindingToken = currentToken;

    const originalColorNode = material.colorNode;
    const originalNormalNode = material.normalNode;
    const originalRoughnessNode = material.roughnessNode;
    const originalMetalnessNode = material.metalnessNode;
    const originalAoNode = material.aoNode;

    let isAttached = true;

    // Attach Base Color
    material.colorNode = vec4(nodes.baseColorLinear, float(1.0));

    // Convert Tangent Normal [-1, 1] to normalMap [0, 1] input convention
    const normalMapEncoded = vec4(nodes.normalTangent.mul(0.5).add(0.5), float(1.0));
    material.normalNode = normalMap(normalMapEncoded);

    // Attach Roughness, Metalness, AO
    material.roughnessNode = nodes.roughness;
    material.metalnessNode = nodes.metalness;
    material.aoNode = nodes.ao;
    material.needsUpdate = true;

    const restore = () => {
        if (!isAttached) return;
        isAttached = false;
        if (material.userData.__activeNeuralBindingToken === currentToken) {
            material.colorNode = originalColorNode;
            material.normalNode = originalNormalNode;
            material.roughnessNode = originalRoughnessNode;
            material.metalnessNode = originalMetalnessNode;
            material.aoNode = originalAoNode;
            delete material.userData.__activeNeuralBindingToken;
            material.needsUpdate = true;
        }
    };

    return {
        restore,
        dispose() {
            restore();
        },
    };
}
