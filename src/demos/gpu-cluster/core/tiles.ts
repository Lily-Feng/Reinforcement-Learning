/**
 * Tile coding, ported from Sutton's tiles3: an index hash table (IHT) plus
 * `tiles` and `tileswrap`. Inputs are floats already scaled so that one unit
 * is one tile width; each tiling is offset by (1, 3, 5, …)/numTilings in the
 * successive dimensions, as tiles3 does.
 */

export interface IHT {
  size: number;
  dict: Map<string, number>;
  overfullCount: number;
}

export function createIHT(size: number): IHT {
  return { size, dict: new Map(), overfullCount: 0 };
}

/** A deterministic 32-bit string hash for the overflow case (tiles3 uses Python's hash). */
function basehash(key: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < key.length; i += 1) h = Math.imul(h ^ key.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** Index for a coordinate tuple; new tuples get the next free index until the table is full. */
export function hashcoords(coords: readonly number[], iht: IHT, readonly = false): number {
  const key = coords.join(',');
  const found = iht.dict.get(key);
  if (found !== undefined) return found;
  if (readonly) return -1;
  const count = iht.dict.size;
  if (count >= iht.size) {
    iht.overfullCount += 1;
    return basehash(key) % iht.size;
  }
  iht.dict.set(key, count);
  return count;
}

export function tiles(iht: IHT, numTilings: number, floats: readonly number[], ints: readonly number[] = [], readonly = false): number[] {
  const qfloats = floats.map(f => Math.floor(f * numTilings));
  const out: number[] = [];
  for (let tiling = 0; tiling < numTilings; tiling += 1) {
    const tilingX2 = tiling * 2;
    const coords = [tiling];
    let b = tiling;
    for (const q of qfloats) {
      coords.push(Math.floor((q + b) / numTilings));
      b += tilingX2;
    }
    coords.push(...ints);
    out.push(hashcoords(coords, iht, readonly));
  }
  return out;
}

/** Like `tiles`, but a dimension with a wrap width w wraps around: coordinate c becomes c mod w. */
export function tileswrap(iht: IHT, numTilings: number, floats: readonly number[], wrapWidths: readonly (number | null)[], ints: readonly number[] = [], readonly = false): number[] {
  const qfloats = floats.map(f => Math.floor(f * numTilings));
  const out: number[] = [];
  for (let tiling = 0; tiling < numTilings; tiling += 1) {
    const tilingX2 = tiling * 2;
    const coords = [tiling];
    let b = tiling;
    qfloats.forEach((q, i) => {
      const width = wrapWidths[i];
      const c = Math.floor((q + (b % numTilings)) / numTilings);
      coords.push(width ? ((c % width) + width) % width : c);
      b += tilingX2;
    });
    coords.push(...ints);
    out.push(hashcoords(coords, iht, readonly));
  }
  return out;
}
