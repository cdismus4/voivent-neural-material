import { describe, expect, it } from 'vitest';
import {
    auditVnmPackageQuality,
    exportNeuralMaterial,
    generateSyntheticPbrTexture,
    NeuralMaterialTrainer,
} from '../index';

describe('PbrQualityAudit', () => {
    it('audits the quantized VNM1 package and returns finite quality metrics', async () => {
        const source = generateSyntheticPbrTexture(16, 16, 'gradient-waves');
        const training = new NeuralMaterialTrainer({
            fineWidth: 8,
            fineHeight: 8,
            coarseWidth: 4,
            coarseHeight: 4,
            batchSize: 32,
            iterations: 2,
        }).train(source);
        const pkg = await exportNeuralMaterial(training, source, { assetId: 'quality-audit-test' });
        const audit = auditVnmPackageQuality(source, pkg, 8);

        expect(audit.sampleCount).toBe(64);
        expect(Number.isFinite(audit.colorMae)).toBe(true);
        expect(Number.isFinite(audit.normalAngularMae)).toBe(true);
        expect(Number.isFinite(audit.overallMae)).toBe(true);
        expect(typeof audit.accepted).toBe('boolean');
    });
});
