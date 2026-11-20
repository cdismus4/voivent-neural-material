/**
 * Analytical PBR Texture Generator for Neural Material Training & Testing
 * Generates float32 8-channel ground truth textures:
 * [0..2]: BaseColor (linear sRGB)
 * [3..4]: Normal tangent (nx, ny in [-1, 1])
 * [5]:    Roughness [0, 1]
 * [6]:    Metalness [0, 1]
 * [7]:    Ambient Occlusion [0, 1]
 */

export interface PbrTextureData {
  readonly width: number;
  readonly height: number;
  /** Flattened array of width * height * 8 elements */
  readonly data: Float32Array;
}

export type PromptPbrPattern = 'auto' | 'smooth' | 'microflake' | 'waves';

export function generateSyntheticPbrTexture(
  width = 256,
  height = 256,
  pattern: 'checker-spheres' | 'gradient-waves' = 'checker-spheres',
): PbrTextureData {
  const data = new Float32Array(width * height * 8);

  for (let y = 0; y < height; y++) {
    const v = (y + 0.5) / height;
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const offset = (y * width + x) * 8;

      if (pattern === 'checker-spheres') {
        // Checkerboard frequency
        const checkU = Math.floor(u * 8) % 2;
        const checkV = Math.floor(v * 8) % 2;
        const isCheck = (checkU ^ checkV) === 1;

        // Local sphere bump inside each cell
        const localU = (u * 8) % 1.0 - 0.5;
        const localV = (v * 8) % 1.0 - 0.5;
        const rSq = localU * localU + localV * localV;
        const inSphere = rSq < 0.16; // radius = 0.4

        let nx = 0.0;
        let ny = 0.0;
        if (inSphere) {
          const r = Math.sqrt(rSq);
          const factor = Math.sin((r / 0.4) * (Math.PI / 2));
          nx = (localU / 0.4) * factor * 0.6;
          ny = (localV / 0.4) * factor * 0.6;
        }

        // BaseColor: Checker colors (Linear sRGB)
        if (isCheck) {
          data[offset + 0] = 0.85; // R
          data[offset + 1] = 0.65; // G
          data[offset + 2] = 0.20; // B (Gold-ish)
        } else {
          data[offset + 0] = 0.15; // R
          data[offset + 1] = 0.25; // G
          data[offset + 2] = 0.45; // B (Slate-ish)
        }

        // Normal
        data[offset + 3] = nx;
        data[offset + 4] = ny;

        // Roughness: rougher on non-sphere
        data[offset + 5] = inSphere ? 0.2 : (isCheck ? 0.4 : 0.8);

        // Metalness: high on gold check
        data[offset + 6] = isCheck ? 0.9 : 0.0;

        // AO: contact occlusion in sphere crevices
        data[offset + 7] = inSphere ? (1.0 - (rSq / 0.16) * 0.3) : 1.0;
      } else {
        // Gradient Waves
        const wave = Math.sin(u * Math.PI * 4) * Math.cos(v * Math.PI * 4);
        data[offset + 0] = 0.5 + 0.5 * Math.sin(u * Math.PI * 2);
        data[offset + 1] = 0.5 + 0.5 * Math.cos(v * Math.PI * 2);
        data[offset + 2] = 0.5 + 0.5 * wave;

        data[offset + 3] = 0.4 * Math.cos(u * Math.PI * 4);
        data[offset + 4] = 0.4 * Math.sin(v * Math.PI * 4);
        data[offset + 5] = 0.3 + 0.4 * Math.abs(wave);
        data[offset + 6] = u > 0.5 ? 0.8 : 0.1;
        data[offset + 7] = 0.7 + 0.3 * (1.0 - Math.abs(wave));
      }
    }
  }

  return { width, height, data };
}

function hashPrompt(prompt: string): number {
  let value = 2166136261;
  for (let index = 0; index < prompt.length; index += 1) {
    value ^= prompt.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clampSignedNormal(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function promptIncludesKeyword(prompt: string, keyword: string): boolean {
  // 英単語を部分一致にすると weathered の末尾 "red" のような誤判定が起きる。
  // 日本語など単語境界を持たない語は従来どおり包含判定にする。
  if (/^[a-z]+$/i.test(keyword)) {
    return new RegExp(`\\b${keyword}\\b`, 'i').test(prompt);
  }
  return prompt.includes(keyword);
}

function promptColor(prompt: string): readonly [number, number, number] {
  const colorTable: Array<{ words: string[]; color: readonly [number, number, number] }> = [
    { words: ['red', 'scarlet', 'crimson', '赤', 'レッド'], color: [0.72, 0.045, 0.025] },
    { words: ['blue', 'navy', '青', 'ブルー'], color: [0.035, 0.16, 0.72] },
    { words: ['green', 'emerald', '緑', 'グリーン'], color: [0.08, 0.42, 0.12] },
    { words: ['gold', 'golden', '金', 'ゴールド'], color: [0.82, 0.48, 0.08] },
    { words: ['silver', 'chrome', '銀', 'シルバー'], color: [0.62, 0.66, 0.7] },
    { words: ['white', 'ivory', '白', 'ホワイト'], color: [0.82, 0.82, 0.78] },
    { words: ['black', '黒', 'ブラック'], color: [0.035, 0.04, 0.045] },
    { words: ['brown', 'wood', '木', '茶'], color: [0.32, 0.12, 0.045] },
  ];
  const match = colorTable.find((entry) => entry.words.some((word) => promptIncludesKeyword(prompt, word)));
  return match?.color ?? [0.42, 0.46, 0.5];
}

/**
 * Creates a prompt-conditioned PBR source for the VNM1 trainer.
 * This is an app-native deterministic source generator, not an external image provider.
 */
export function generatePromptPbrTexture(
  width = 64,
  height = 64,
  prompt = '',
  pattern: PromptPbrPattern = 'auto',
): PbrTextureData {
  const safeWidth = Math.max(1, Math.round(width));
  const safeHeight = Math.max(1, Math.round(height));
  const normalizedPrompt = prompt.toLowerCase();
  const seed = hashPrompt(normalizedPrompt);
  const base = promptColor(normalizedPrompt);
  const metallic = /metal|chrome|steel|car paint|金属|クローム|車塗装/.test(normalizedPrompt);
  const glossy = /gloss|glossy|paint|polished|lacquer|光沢|塗装|鏡面/.test(normalizedPrompt);
  const organic = /wood|bark|leaf|moss|grass|木|樹皮|葉|苔|草/.test(normalizedPrompt);
  const stone = /stone|rock|concrete|砂岩|石|岩|コンクリート/.test(normalizedPrompt);
  const mentionsPattern = /pattern|stripe|striped|weave|woven|checker|縞|ストライプ|織物|格子|模様/.test(normalizedPrompt);
  const mentionsExclusion = /no\b|without|exclude|excluding|avoid|除外|なし|無し|不要|含めない|入れない|避け/.test(normalizedPrompt);
  const useDirectionalPattern = pattern === 'waves' || (pattern === 'auto' && mentionsPattern && !mentionsExclusion);
  const useMicroflake = pattern === 'microflake' || (pattern === 'auto' && metallic && !useDirectionalPattern);
  const data = new Float32Array(safeWidth * safeHeight * 8);

  for (let y = 0; y < safeHeight; y += 1) {
    const v = (y + 0.5) / safeHeight;
    for (let x = 0; x < safeWidth; x += 1) {
      const u = (x + 0.5) / safeWidth;
      const offset = (y * safeWidth + x) * 8;
      const phase = (seed % 997) * 0.001;
      const directionalWave = Math.sin((u * 9.0 + phase) * Math.PI) * Math.cos((v * 7.0 - phase) * Math.PI);
      const isotropicWave = Math.sin((u * 11.17 + v * 17.31 + phase) * Math.PI) * Math.cos((u * 6.13 - v * 8.71 - phase) * Math.PI);
      // smooth は塗膜や無地のような「パターン無し」の明示指定。以前はこの
      // 経路にも波と固定の縦方向法線が混ざり、圧縮後に縞として現れていた。
      const wave = pattern === 'smooth'
        ? 0
        : useDirectionalPattern
          ? directionalWave
          : isotropicWave * 0.22;
      const fine = useMicroflake
        ? Math.sin((u * 73.7 + v * 91.1 + phase) * Math.PI) * Math.cos((u * 47.3 - v * 61.7 + phase) * Math.PI)
        : pattern === 'smooth'
          ? 0
          : Math.sin((u * 37.0 + v * 29.0 + phase) * Math.PI);
      const textureVariation = wave * 0.08 + fine * 0.025;
      const heightValue = clamp01(0.5 + wave * 0.22 + fine * 0.06);
      const colorScale = 1 + textureVariation;

      data[offset] = clamp01(base[0] * colorScale);
      data[offset + 1] = clamp01(base[1] * colorScale);
      data[offset + 2] = clamp01(base[2] * colorScale);
      // PBR source の tangent normal は [-1, 1]。周期的な cos/sin を無条件で
      // 加えず、実際の生成パターンにだけ小さく追従させる。
      const normalDetail = pattern === 'smooth' ? 0 : wave * 0.16 + fine * 0.035;
      data[offset + 3] = clampSignedNormal(normalDetail);
      data[offset + 4] = clampSignedNormal(useDirectionalPattern ? directionalWave * 0.12 : fine * 0.03);
      data[offset + 5] = clamp01(glossy ? 0.18 + heightValue * 0.18 : stone ? 0.68 + heightValue * 0.2 : organic ? 0.58 + heightValue * 0.25 : 0.42 + heightValue * 0.25);
      data[offset + 6] = metallic ? 0.78 + heightValue * 0.18 : 0.02;
      data[offset + 7] = clamp01(0.82 + (1 - heightValue) * 0.16);
    }
  }

  return { width: safeWidth, height: safeHeight, data };
}
