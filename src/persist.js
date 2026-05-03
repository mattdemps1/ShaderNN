// MLP weight persistence: raw little-endian binary blob with header.
//
// Layout:
//   uint32 magic   = 'SNN2' (0x32_4E_4E_53 LE)
//   uint32 version = 2
//   uint32 D, H, C
//   float32 W1[H*D], b1[H], W2[C*H], b2[C]
//
// Why not PNG: encoding floats as RGBA8 round-trips through Canvas2D's
// premultiplied-alpha storage, which corrupts bytes whenever alpha < 255.
// .bin via fetch / Blob is lossless and trivial.
//
// Auto-persistence uses IndexedDB (localStorage's 5MB cap can't hold the
// MLP weights for D≈33k, H=64).

const MAGIC = 0x324E4E53;  // 'SNN2' little-endian
const VERSION = 2;
const DB_NAME = "shadernn";
const STORE = "weights";
const KEY = "readout";

function pack(readout) {
  const { W1, b1, W2, b2, D, H, C } = readout;
  const headerWords = 5;  // magic, version, D, H, C
  const totalFloats = W1.length + b1.length + W2.length + b2.length;
  const buf = new ArrayBuffer(headerWords * 4 + totalFloats * 4);
  const dv = new DataView(buf);
  dv.setUint32(0,  MAGIC,   true);
  dv.setUint32(4,  VERSION, true);
  dv.setUint32(8,  D,       true);
  dv.setUint32(12, H,       true);
  dv.setUint32(16, C,       true);
  const floats = new Float32Array(buf, headerWords * 4);
  let off = 0;
  floats.set(W1, off); off += W1.length;
  floats.set(b1, off); off += b1.length;
  floats.set(W2, off); off += W2.length;
  floats.set(b2, off);
  return buf;
}

function unpack(buf, readout) {
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  if (magic !== MAGIC) throw new Error(`bad magic 0x${magic.toString(16)} (need SNN2 .bin)`);
  const version = dv.getUint32(4, true);
  if (version !== VERSION) throw new Error(`unsupported version ${version}`);
  const D = dv.getUint32(8, true);
  const H = dv.getUint32(12, true);
  const C = dv.getUint32(16, true);
  if (D !== readout.D || H !== readout.H || C !== readout.C) {
    throw new Error(`shape mismatch: file ${D}×${H}×${C}, readout ${readout.D}×${readout.H}×${readout.C}`);
  }
  const floats = new Float32Array(buf, 5 * 4);
  let off = 0;
  readout.W1.set(floats.subarray(off, off + readout.W1.length)); off += readout.W1.length;
  readout.b1.set(floats.subarray(off, off + readout.b1.length)); off += readout.b1.length;
  readout.W2.set(floats.subarray(off, off + readout.W2.length)); off += readout.W2.length;
  readout.b2.set(floats.subarray(off, off + readout.b2.length));
}

export function saveBin(readout, filename = "shadernn-weights.bin") {
  const buf = pack(readout);
  const blob = new Blob([buf], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export async function loadBin(file, readout) {
  const buf = await file.arrayBuffer();
  unpack(buf, readout);
}

export async function loadDefaultBin(readout, url = "./data/default-weights.bin") {
  try {
    const r = await fetch(url);
    if (!r.ok) return false;
    const buf = await r.arrayBuffer();
    unpack(buf, readout);
    return true;
  } catch (e) {
    console.warn(`loadDefaultBin: ${e.message}`);
    return false;
  }
}

// IndexedDB helpers — async, no quota issue for our ~17MB blob.
function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

export async function saveLocal(readout) {
  try {
    const db = await openDB();
    const buf = pack(readout);
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(buf, KEY);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
    db.close();
    return true;
  } catch (e) {
    console.warn("saveLocal failed:", e.message);
    return false;
  }
}

export async function loadLocal(readout) {
  try {
    const db = await openDB();
    const buf = await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readonly");
      const req = tx.objectStore(STORE).get(KEY);
      req.onsuccess = () => resolve(req.result);
      req.onerror   = () => reject(req.error);
    });
    db.close();
    if (!buf) return false;
    unpack(buf, readout);
    return true;
  } catch (e) {
    console.warn("loadLocal failed:", e.message);
    return false;
  }
}

export async function clearLocal() {
  try {
    const db = await openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).delete(KEY);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (_) {}
}
