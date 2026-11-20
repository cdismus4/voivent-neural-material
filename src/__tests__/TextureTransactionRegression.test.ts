import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

interface RequestStub {
    result: unknown;
    onsuccess?: () => void;
    onerror?: () => void;
}
interface TransactionStub {
    oncomplete?: () => void;
    onabort?: () => void;
    onerror?: () => void;
    error: Error | null;
}

function installDatabase() {
    const transactions: TransactionStub[] = [];
    const requests: RequestStub[] = [];
    const db = {
        objectStoreNames: { contains: () => true },
        transaction: () => {
            const request = (result: unknown) => {
                const req: RequestStub = { result };
                requests.push(req);
                return req;
            };
            const tx = { error: null, oncomplete: undefined, onabort: undefined, onerror: undefined,
                objectStore: () => ({ getAll: () => request([]), get: () => request(undefined), count: () => request(1), put: () => request(undefined),
                    delete: () => request(undefined), clear: () => request(undefined) }) } satisfies TransactionStub & { objectStore: () => unknown };
            transactions.push(tx);
            return tx;
        },
    };
    vi.stubGlobal('indexedDB', { open: () => {
        const req: RequestStub = { result: db };
        queueMicrotask(() => req.onsuccess?.());
        return req;
    } });
    vi.stubGlobal('IDBKeyRange', { only: (key: string) => key });
    return { transactions, requests };
}

describe('persistent transaction completion', () => {
    beforeEach(() => { vi.resetModules(); });
    afterEach(() => { vi.unstubAllGlobals(); });

    it.each(['delete', 'clear'] as const)('waits for cache %s commit, not just request success', async (operation) => {
        const db = installDatabase();
        const cache = await import('../runtime/PersistentChunkCache');
        let settled = false;
        const action = (operation === 'delete' ? cache.deleteCachedChunk('id') : cache.clearChunkCache()).then(() => { settled = true; });
        await vi.waitFor(() => expect(db.transactions).toHaveLength(1));
        db.requests[0].onsuccess?.();
        await Promise.resolve();
        expect(settled).toBe(false);
        db.transactions[0].oncomplete?.();
        await action;
        expect(settled).toBe(true);
    });
});
