# Voivent Neural Material (`voivent-neural-material`)

TypeScript & Three.js **WebGPU / TSL** runtime, progressive streaming loader, conformance verifier, and trainer for **VNM1 (Voivent Neural Material)** PBR materials.

## Requirements

> **Important — WebGPU Only:**  
> Real-time GPU decoding requires **`three` (>= 0.185.0) using `WebGPURenderer` (`three/webgpu`) and Three Shading Language (`three/tsl`)**.  
> The production material binder attaches neural PBR node graphs (`MeshStandardNodeMaterial`) directly on the GPU with **no WebGL / CPU runtime fallback**. Classic `WebGLRenderer` is not supported for real-time rendering.

- Modern browser or runtime with WebGPU support (`WebGPURenderer` / WGSL)
- Peer dependency: `three >= 0.185.0 < 1.0.0`

## Profiles

| Profile | Input Dim | Hidden Width | Latent Ch / Grid | Default Latent Sizes (Trainer Preset) | Total Weights | MACs / Texel |
|---|---:|---:|---:|---|---:|---:|
| `small` | 12 | 16 / 16 | 4 | custom | 616 | 576 |
| `standard` | 12 | 24 / 24 | 4 | 512x512 / 256x256 | 1,112 | 1,056 |
| `standard8` | 20 | 24 / 24 | 8 | 512x512 / 256x256 | 1,304 | 1,248 |
| `quality` | 12 | 32 / 32 | 4 | 1024x1024 / 512x512 | 1,736 | 1,664 |
| `quality8` | 20 | 32 / 32 | 8 | 1024x1024 / 512x512 | 1,992 | 1,920 |

See [`docs/SPEC.md`](./docs/SPEC.md) for the complete binary format & conformance specification and [`docs/CAPABILITY.md`](./docs/CAPABILITY.md) for measured quality and desktop/mobile GPU performance characteristics.

## Quickstart (Streaming + Direct WebGPU Binding)

```ts
import * as THREE from 'three/webgpu';
import {
  bindResidentAssetToMeshes,
  parseManifest,
  unbindResidentAssetFromMeshes,
  VnmChunkLoader,
  VnmResidencyManager,
  VnmStreamingSession,
} from 'voivent-neural-material';

const manifest = parseManifest(await (await fetch('/assets/marble/manifest.json')).json());
const loader = new VnmChunkLoader();
const residency = new VnmResidencyManager();

const mesh = new THREE.Mesh(
  new THREE.SphereGeometry(1, 64, 64),
  new THREE.MeshStandardNodeMaterial(),
);

const session = new VnmStreamingSession(loader, residency, {
  onTierChanged: (_tier, state) => {
    if (state !== 'active') return;
    const asset = session.getCurrentAsset();
    if (!asset) return;
    // Automatically shows tail DataTexture preview first, then hot-swaps to
    // TslMlpDecoderNode fragment decoding on level0 completion (disposing prior GPU resources).
    bindResidentAssetToMeshes([mesh], asset);
  },
});

await session.loadMaterial(manifest);

// Later cleanup: restores original material and disposes GPU textures/bindings
unbindResidentAssetFromMeshes([mesh]);
```

## Module Overview

- `format/` — VNM1 manifest parser/validator (`parseManifest`, `resolveManifestMipMax`) and binary chunk parser/serializer (`weights.bin`, `level-0.bin`, `tail.bin`) with SHA-256 verification metadata.
- `gpu/` — Pure TSL fragment decoder (`createNeuralMaterialTslNodes`, `attachNeuralPbrNodes`), portable Three.js WebGPU binder (`bindResidentAssetToMeshes`, `createTailPreviewTextures`), and WebGPU readback conformance harness (`GpuReadbackVerification`).
- `streaming/` — Progressive `tail -> level0` asset loader (`VnmChunkLoader`), memory-budgeted residency manager (`VnmResidencyManager`), and cancellation-safe `VnmStreamingSession`.
- `reference/` — Canonical CPU float32/double reference decoder (`CpuReferenceDecoder`) for conformance verification against golden vectors (`SPEC §7.1`).
- `exporter/`, `training/`, `runtime/PersistentChunkCache` — VNM1 chunk exporter, TypeScript trainer & quality audit utilities, and IndexedDB chunk persistence.

## Conformance & Tests

```bash
npm run typecheck
npm test
npm run build
```

Includes the 286-case golden vector conformance suites across `standard`, `quality`, `standard8`, and `quality8` in [`src/__tests__/`](./src/__tests__/).

## Known Limitations (VNM1 v1)

- **No intermediate mipmaps / anisotropic filtering in v1**: VNM1 v1 assets train and evaluate mip 0 (`level-0.bin`) plus a coarse `16x16` `tail.bin` fallback without mip 1..6 levels. High-frequency periodic patterns (such as woven fabrics like Denim or fine gravel) will exhibit spatial minification moire when viewed at a distance compared to hardware-mipmapped PBR textures.
- **No alpha cutout, emissive, or height channels**: Standard V1 outputs 8 PBR channels (`baseColor` RGB, tangent `normal` XY, `roughness`, `metalness`, `ao`).
- See [`docs/CAPABILITY.md`](./docs/CAPABILITY.md) for measured corpus bounds (`30/29` pass rate, 4K grid-resolution wall, and desktop/mobile GPU timings).

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).
