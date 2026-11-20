import { describe, expect, it } from 'vitest';
import { generatePromptPbrTexture } from '../training/SyntheticTextureGenerator';

describe('generatePromptPbrTexture', () => {
  it('keeps an explicitly smooth source free of periodic texture and normal patterns', () => {
    const source = generatePromptPbrTexture(16, 16, 'deep red glossy car paint', 'smooth');
    const first = Array.from(source.data.slice(0, 8));

    for (let offset = 0; offset < source.data.length; offset += 8) {
      expect(Array.from(source.data.slice(offset, offset + 3))).toEqual(first.slice(0, 3));
      expect(source.data[offset + 3]).toBe(0);
      expect(source.data[offset + 4]).toBe(0);
    }
  });

  it('writes tangent normal channels in the documented signed range', () => {
    const source = generatePromptPbrTexture(32, 32, 'brushed metal surface', 'waves');

    for (let offset = 0; offset < source.data.length; offset += 8) {
      expect(source.data[offset + 3]).toBeGreaterThanOrEqual(-1);
      expect(source.data[offset + 3]).toBeLessThanOrEqual(1);
      expect(source.data[offset + 4]).toBeGreaterThanOrEqual(-1);
      expect(source.data[offset + 4]).toBeLessThanOrEqual(1);
    }
  });

  it('does not treat the end of weathered as the red color keyword', () => {
    const source = generatePromptPbrTexture(1, 1, 'matte weathered concrete', 'smooth');

    expect(source.data[0]).toBeCloseTo(0.42, 6);
    expect(source.data[1]).toBeCloseTo(0.46, 6);
    expect(source.data[2]).toBeCloseTo(0.5, 6);
  });
});
