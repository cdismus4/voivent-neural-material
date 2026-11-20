/**
 * Voivent Neural Material (VNM1) - Persistent Chunk Cache
 *
 * Implements 2026 browser storage best practices:
 * - High-capacity IndexedDB binary storage (GBs capacity, structured clone for ArrayBuffer).
 * - Zero Base64 string encoding overhead.
 * - Keyed by SHA-256 content hash or chunk URL.
 * - StorageManager persistence request (navigator.storage.persist) to prevent browser eviction.
 */

const DB_NAME = 'voivent-neural-material-cache';
const DB_VERSION = 1;
const STORE_NAME = 'chunks';

export interface CachedChunkRecord {
    readonly key: string;
    readonly data: ArrayBuffer;
    readonly byteLength: number;
    readonly createdAt: number;
    readonly lastAccessedAt: number;
}

export interface CacheStats {
    readonly totalChunks: number;
    readonly totalBytes: number;
}

let dbPromise: Promise<IDBDatabase> | null = null;

function getDatabase(): Promise<IDBDatabase> {
    if (dbPromise) return dbPromise;
    if (typeof indexedDB === 'undefined') {
        return Promise.reject(new Error('IndexedDB is not supported in this environment.'));
    }

    dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME, { keyPath: 'key' });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => {
            dbPromise = null;
            reject(req.error ?? new Error('Failed to open neural material cache database'));
        };
    });

    return dbPromise;
}

/**
 * Requests persistent storage from the browser to prevent automatic cache eviction under disk pressure.
 */
export async function requestStoragePersistence(): Promise<boolean> {
    if (typeof navigator !== 'undefined' && navigator.storage?.persist) {
        try {
            return await navigator.storage.persist();
        } catch {
            return false;
        }
    }
    return false;
}

/**
 * Retrieves a cached binary chunk by key (SHA-256 hash or URL).
 */
async function updateAccessTime(key: string, db: IDBDatabase): Promise<void> {
    return new Promise<void>((resolve) => {
        try {
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            const req = store.get(key);
            req.onsuccess = () => {
                const record = req.result as CachedChunkRecord | undefined;
                if (record) {
                    const updated: CachedChunkRecord = { ...record, lastAccessedAt: Date.now() };
                    store.put(updated);
                }
            };
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
            tx.onabort = () => resolve();
        } catch {
            resolve();
        }
    });
}

export type Sha256VerificationResult = 'match' | 'mismatch' | 'unavailable';

/**
 * Validates whether the given binary data matches an expected SHA-256 hex string.
 * Strictly distinguishes between 'match', 'mismatch', and 'unavailable' (when Web Crypto is missing).
 */
export async function verifyChunkSha256Status(data: Uint8Array, expectedSha256: string): Promise<Sha256VerificationResult> {
    if (typeof crypto === 'undefined' || !crypto.subtle?.digest) {
        return 'unavailable';
    }
    try {
        const hashBuf = await crypto.subtle.digest('SHA-256', data as Uint8Array<ArrayBuffer>);
        const hashArray = Array.from(new Uint8Array(hashBuf));
        const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
        return hashHex.toLowerCase() === expectedSha256.toLowerCase() ? 'match' : 'mismatch';
    } catch {
        return 'unavailable';
    }
}

/**
 * Validates whether the given binary data matches an expected SHA-256 hex string.
 * Returns true ONLY on verified match, false on mismatch or verification failure.
 */
export async function verifyChunkSha256(data: Uint8Array, expectedSha256: string): Promise<boolean> {
    const status = await verifyChunkSha256Status(data, expectedSha256);
    return status === 'match';
}

/**
 * Retrieves a cached binary chunk by key (SHA-256 hash or URL).
 */
export async function getCachedChunk(key: string): Promise<Uint8Array | null> {
    try {
        const db = await getDatabase();
        return await new Promise<Uint8Array | null>((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const req = store.get(key);

            req.onsuccess = async () => {
                const record = req.result as CachedChunkRecord | undefined;
                if (!record || !record.data) {
                    resolve(null);
                    return;
                }
                await updateAccessTime(key, db);
                resolve(new Uint8Array(record.data));
            };
            req.onerror = () => reject(req.error);
        });
    } catch {
        return null;
    }
}

/**
 * Stores a binary chunk into IndexedDB without any Base64 encoding.
 * Requires expectedSha256, copies buffer prior to verification, and resolves strictly on tx.oncomplete.
 */
export async function putCachedChunk(key: string, data: Uint8Array, expectedSha256: string): Promise<void> {
    if (!expectedSha256 || typeof expectedSha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(expectedSha256)) {
        throw new Error(`expectedSha256 is required for storing delivery chunk '${key}' and must be 64 hex characters`);
    }

    // 1. Copy data to an immutable dedicated buffer FIRST before asynchronous verification
    const buffer = new ArrayBuffer(data.byteLength);
    const copiedData = new Uint8Array(buffer);
    copiedData.set(data);

    // 2. Verify hash against the exact copied buffer
    const status = await verifyChunkSha256Status(copiedData, expectedSha256);
    if (status === 'mismatch') {
        throw new Error(`SHA-256 hash mismatch for chunk ${key}`);
    }
    if (status === 'unavailable') {
        throw new Error(`SHA-256 verification unavailable: crypto.subtle is not supported in this environment`);
    }

    const db = await getDatabase();
    const record: CachedChunkRecord = {
        key,
        data: buffer,
        byteLength: buffer.byteLength,
        createdAt: Date.now(),
        lastAccessedAt: Date.now(),
    };

    return new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        store.put(record);

        tx.oncomplete = () => resolve();
        tx.onabort = () => reject(tx.error ?? new Error(`Transaction aborted while storing chunk ${key}`));
        tx.onerror = () => reject(tx.error ?? new Error(`Failed to store chunk ${key}`));
    });
}

/**
 * Checks whether a chunk is available in local cache.
 */
export async function hasCachedChunk(key: string): Promise<boolean> {
    try {
        const db = await getDatabase();
        return await new Promise<boolean>((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const req = store.count(IDBKeyRange.only(key));
            req.onsuccess = () => resolve(req.result > 0);
            req.onerror = () => reject(req.error);
        });
    } catch {
        return false;
    }
}

/**
 * Removes a specific chunk from the cache.
 */
export async function deleteCachedChunk(key: string): Promise<void> {
    try {
        const db = await getDatabase();
        return await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            store.delete(key);
            tx.oncomplete = () => resolve();
            tx.onabort = () => reject(tx.error ?? new Error('Cache delete transaction aborted'));
            tx.onerror = () => reject(tx.error ?? new Error('Cache delete transaction failed'));
        });
    } catch {
        // Safe no-op on failure
    }
}

/**
 * Clears the entire binary chunk cache.
 */
export async function clearChunkCache(): Promise<void> {
    try {
        const db = await getDatabase();
        return await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readwrite');
            const store = tx.objectStore(STORE_NAME);
            store.clear();
            tx.oncomplete = () => resolve();
            tx.onabort = () => reject(tx.error ?? new Error('Cache clear transaction aborted'));
            tx.onerror = () => reject(tx.error ?? new Error('Cache clear transaction failed'));
        });
    } catch {
        // Safe no-op on failure
    }
}

/**
 * Calculates current cache statistics (total chunks and bytes).
 */
export async function getCacheStats(): Promise<CacheStats> {
    try {
        const db = await getDatabase();
        return await new Promise<CacheStats>((resolve, reject) => {
            const tx = db.transaction(STORE_NAME, 'readonly');
            const store = tx.objectStore(STORE_NAME);
            const req = store.openCursor();
            let totalChunks = 0;
            let totalBytes = 0;

            req.onsuccess = () => {
                const cursor = req.result;
                if (cursor) {
                    const record = cursor.value as CachedChunkRecord;
                    totalChunks++;
                    totalBytes += record.byteLength ?? 0;
                    cursor.continue();
                } else {
                    resolve({ totalChunks, totalBytes });
                }
            };
            req.onerror = () => reject(req.error);
        });
    } catch {
        return { totalChunks: 0, totalBytes: 0 };
    }
}
