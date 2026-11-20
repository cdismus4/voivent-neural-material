/**
 * Neural Material Loss Functions & Exact Analytical Gradients
 * Strictly adheres to NTC_PLAN.md specification:
 * L = 4 * L_color + 2 * L_normal + L_roughness + L_metalness + L_ao
 */

export interface LossResult {
  totalLoss: number;
  colorLoss: number;
  normalLoss: number;
  roughnessLoss: number;
  metalnessLoss: number;
  aoLoss: number;
  /** Exact analytical gradient dL/d(pred) of length 8 */
  gradPred: Float32Array;
}

/**
 * Reconstructs a tangent-space unit normal from predicted (px, py).
 * Follows the exact production shader formula in TslMlpDecoderNode and CpuReferenceDecoder:
 * 1. Clamp px, py to [-1, 1]
 * 2. nz = sqrt(max(0, 1 - px^2 - py^2))
 * 3. normalize(nx, ny, nz)
 */
export function reconstructNormal(px: number, py: number): [number, number, number] {
  const nx = Math.max(-1.0, Math.min(1.0, px));
  const ny = Math.max(-1.0, Math.min(1.0, py));
  const xySq = nx * nx + ny * ny;
  const nz = Math.sqrt(Math.max(0.0, 1.0 - xySq));
  const len = Math.hypot(nx, ny, nz);
  return len < 1e-12 ? [0, 0, 1] : [nx / len, ny / len, nz / len];
}

/**
 * Computes exact multi-task PBR loss and unclipped analytical gradients.
 */
export function computeMaterialLoss(
  pred: ArrayLike<number>,
  target: ArrayLike<number>,
): LossResult {
  // 1. Color Loss (MSE across 3 channels, weight = 4.0)
  const dR = pred[0] - target[0];
  const dG = pred[1] - target[1];
  const dB = pred[2] - target[2];
  const colorMse = (dR * dR + dG * dG + dB * dB) / 3.0;
  const colorLoss = 4.0 * colorMse;

  // 2. Normal Loss (Angular loss: 2 * (1 - n · t), weight = 2.0)
  // Target normal (tx, ty, tz) is a unit tangent normal
  const [tx, ty, tz] = reconstructNormal(target[3], target[4]);

  const px = pred[3];
  const py = pred[4];
  const [nx, ny, nz] = reconstructNormal(px, py);

  // Cosine similarity
  const dot = nx * tx + ny * ty + nz * tz;
  const normalLoss = 2.0 * Math.max(0.0, 1.0 - dot);

  // Exact analytical derivative of L_normal with respect to px and py
  let gradNx = 0.0;
  let gradNy = 0.0;

  if (normalLoss > 0.0) {
    const clPx = Math.max(-1.0, Math.min(1.0, px));
    const clPy = Math.max(-1.0, Math.min(1.0, py));
    const xySq = clPx * clPx + clPy * clPy;

    if (xySq < 1.0) {
      // In-disk case: nz = sqrt(1 - px^2 - py^2)
      // dL/dpx = 2.0 * (tz * px / nz - tx)
      const safeNz = Math.sqrt(1.0 - xySq);
      gradNx = 2.0 * ((tz * clPx) / safeNz - tx);
      gradNy = 2.0 * ((tz * clPy) / safeNz - ty);
    } else {
      // Boundary / outside disk case: r = sqrt(px^2 + py^2), nz = 0
      // L = 2 * (1 - (px*tx + py*ty) / r)
      // dL/dpx = 2 * (dot2D * px / r^3 - tx / r) = 2 * (dot2D * px / r^2 - tx) / r
      const r = Math.sqrt(xySq);
      const invR = 1.0 / r;
      const invR2 = invR * invR;
      const dot2D = clPx * tx + clPy * ty;
      gradNx = 2.0 * ((dot2D * clPx * invR2 - tx) * invR);
      gradNy = 2.0 * ((dot2D * clPy * invR2 - ty) * invR);
    }
  }

  // Chain rule through component clamping. At the non-differentiable clamp endpoints,
  // select zero; at the circle itself use the finite outside-branch derivative above.
  if (px <= -1 || px >= 1) gradNx = 0;
  if (py <= -1 || py >= 1) gradNy = 0;

  // 3. Roughness Loss (MSE, weight = 1.0)
  const dRough = pred[5] - target[5];
  const roughnessLoss = dRough * dRough;

  // 4. Metalness Loss (MSE, weight = 1.0)
  const dMetal = pred[6] - target[6];
  const metalnessLoss = dMetal * dMetal;

  // 5. AO Loss (MSE, weight = 1.0)
  const dAo = pred[7] - target[7];
  const aoLoss = dAo * dAo;

  const totalLoss = colorLoss + normalLoss + roughnessLoss + metalnessLoss + aoLoss;

  const gradPred = new Float32Array(8);
  // Color grad: d(4 * 1/3 * sum(d^2)) / d(pred) = 8/3 * d
  gradPred[0] = (8.0 / 3.0) * dR;
  gradPred[1] = (8.0 / 3.0) * dG;
  gradPred[2] = (8.0 / 3.0) * dB;

  // Normal grad: true analytical gradient without hard clip inside loss
  gradPred[3] = gradNx;
  gradPred[4] = gradNy;

  // Roughness, Metalness, AO grad: d(d^2) / d(pred) = 2 * d
  gradPred[5] = 2.0 * dRough;
  gradPred[6] = 2.0 * dMetal;
  gradPred[7] = 2.0 * dAo;

  return {
    totalLoss,
    colorLoss,
    normalLoss,
    roughnessLoss,
    metalnessLoss,
    aoLoss,
    gradPred,
  };
}
