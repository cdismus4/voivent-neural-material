import { describe, expect, it } from 'vitest';
import {
    clearChunkCache,
    deleteCachedChunk,
    getCachedChunk,
    getCacheStats,
    hasCachedChunk,
    putCachedChunk,
    requestStoragePersistence,
} from '../runtime/PersistentChunkCache';

describe('PersistentChunkCache', () => {
    it('handles environment gracefully when indexedDB is undefined', async () => {
        const key = 'test-chunk-123';
        const chunk = await getCachedChunk(key);
        expect(chunk).toBeNull();

        const has = await hasCachedChunk(key);
        expect(has).toBe(false);

        const stats = await getCacheStats();
        expect(stats).toEqual({ totalChunks: 0, totalBytes: 0 });

        // Safe no-op delete and clear
        await expect(deleteCachedChunk(key)).resolves.not.toThrow();
        await expect(clearChunkCache()).resolves.not.toThrow();

        // requestStoragePersistence returns false when navigator.storage is undefined
        const persisted = await requestStoragePersistence();
        expect(typeof persisted).toBe('boolean');
    });

    it('stores and retrieves binary chunks when indexedDB mock is provided', async () => {
        // Mock a lightweight in-memory IDB implementation for Node.js test environment
        const memoryStore = new Map<string, unknown>();
        const mockDb = {
            transaction: () => {
                const tx = {
                    objectStore: () => ({
                        get: (k: string) => {
                            const req: { result?: unknown; onsuccess?: () => void; onerror?: () => void } = {
                                result: memoryStore.get(k),
                            };
                            setTimeout(() => req.onsuccess?.(), 0);
                            return req;
                        },
                        put: (record: { key: string }) => {
                            memoryStore.set(record.key, record);
                            const req: { onsuccess?: () => void; onerror?: () => void } = {};
                            setTimeout(() => req.onsuccess?.(), 0);
                            return req;
                        },
                        delete: (k: string) => {
                            memoryStore.delete(k);
                            const req: { onsuccess?: () => void; onerror?: () => void } = {};
                            setTimeout(() => req.onsuccess?.(), 0);
                            return req;
                        },
                        count: () => {
                            const req: { result: number; onsuccess?: () => void; onerror?: () => void } = {
                                result: memoryStore.size,
                            };
                            setTimeout(() => req.onsuccess?.(), 0);
                            return req;
                        },
                        clear: () => {
                            memoryStore.clear();
                            const req: { onsuccess?: () => void; onerror?: () => void } = {};
                            setTimeout(() => req.onsuccess?.(), 0);
                            return req;
                        },
                    }),
                    oncomplete: null as (() => void) | null,
                    onerror: null as (() => void) | null,
                    onabort: null as (() => void) | null,
                };
                setTimeout(() => tx.oncomplete?.(), 5);
                return tx;
            },
            objectStoreNames: { contains: () => true },
        };

        const originalIdb = globalThis.indexedDB;
        try {
            (globalThis as unknown as { indexedDB: unknown }).indexedDB = {
                open: () => {
                    const req: { result: typeof mockDb; onsuccess?: () => void; onerror?: () => void } = {
                        result: mockDb,
                    };
                    setTimeout(() => req.onsuccess?.(), 0);
                    return req;
                },
            };

            const testKey = 'chunk-sha256-abcdef123456';
            const testPayload = new Uint8Array([1, 2, 3, 4, 5, 42, 255]);
            const hashBuf = await crypto.subtle.digest('SHA-256', testPayload);
            const expectedSha256 = Array.from(new Uint8Array(hashBuf))
                .map((b) => b.toString(16).padStart(2, '0'))
                .join('');

            await putCachedChunk(testKey, testPayload, expectedSha256);
            const retrieved = await getCachedChunk(testKey);

            expect(retrieved).not.toBeNull();
            expect(retrieved?.length).toBe(testPayload.length);
            expect(retrieved?.[5]).toBe(42);
            expect(retrieved?.[6]).toBe(255);

            await deleteCachedChunk(testKey);
            const deleted = await getCachedChunk(testKey);
            expect(deleted).toBeNull();
        } finally {
            (globalThis as unknown as { indexedDB: unknown }).indexedDB = originalIdb;
        }
    });
});
