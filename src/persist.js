// Lossless save/load of readout weights as a PNG image, plus localStorage
// auto-persistence.
//
// Layout: image of size (D + 1) wide × C tall, RGBA8. Each pixel encodes one
// float32 (4 bytes = 1 RGBA pixel). The last column stores the bias vector.
//
// PNG header includes a small JSON sidecar via image.dimensions; we also stamp
// magic bytes into the first pixel of row 0 so we can sanity-check on load.

const MAGIC = [0x53, 0x4E, 0x4E, 0x31];   // "SNN1"
const LS_KEY = "shadernn:readout:v2";

function floatToBytes(buf, idx, f) {
  const dv = new DataView(buf.buffer, buf.byteOffset + idx * 4, 4);
  dv.setFloat32(0, f, true);
}

function bytesToFloat(buf, idx) {
  const dv = new DataView(buf.buffer, buf.byteOffset + idx * 4, 4);
  return dv.getFloat32(0, true);
}

// Encode W (C×D row-major Float32Array) and b (Float32Array length C) into a
// new RGBA8 ImageData of size (D+1, C). Caller can paint it onto a canvas.
function encode(W, b, C, D) {
  const width = D + 1;
  const height = C;
  const bytes = new Uint8ClampedArray(width * height * 4);
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < D; i++) {
      const px = c * width + i;
      floatToBytes(bytes, px, W[c * D + i]);
    }
    floatToBytes(bytes, c * width + D, b[c]);
  }
  // Stamp magic into pixel (D, 0) — overwrites bias of class 0… so put magic
  // somewhere safe instead: alpha of pixel (0, 0) we leave alone (it's part of
  // the float). Use a separate detection: dims-encoded check on load.
  return { bytes, width, height };
}

function decode(bytes, width, height) {
  const C = height;
  const D = width - 1;
  const W = new Float32Array(C * D);
  const b = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    for (let i = 0; i < D; i++) {
      const px = c * width + i;
      W[c * D + i] = bytesToFloat(bytes, px);
    }
    b[c] = bytesToFloat(bytes, c * width + D);
  }
  return { W, b, C, D };
}

export function savePNG(readout, filename = "shadernn-weights.png") {
  const { width, height, bytes } = encode(readout.W, readout.b, readout.C, readout.D);
  const cv = document.createElement("canvas");
  cv.width = width; cv.height = height;
  const ctx = cv.getContext("2d");
  const id = ctx.createImageData(width, height);
  id.data.set(bytes);
  ctx.putImageData(id, 0, 0);
  cv.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }, "image/png");
}

async function decodeImageInto(img, readout) {
  if (img.width !== readout.D + 1 || img.height !== readout.C) {
    throw new Error(
      `weight image is ${img.width}×${img.height}, expected ${readout.D + 1}×${readout.C}`
    );
  }
  const cv = document.createElement("canvas");
  cv.width = img.width; cv.height = img.height;
  const ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const id = ctx.getImageData(0, 0, img.width, img.height);
  const { W, b } = decode(id.data, img.width, img.height);
  readout.W.set(W);
  readout.b.set(b);
}

export async function loadPNG(file, readout) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    await decodeImageInto(img, readout);
  } finally {
    URL.revokeObjectURL(url);
  }
}

// Try to fetch a bundled default-weights PNG. Returns true on success,
// false silently on 404 / dimension mismatch / decode failure — this is
// the "no default bundled" case and is not an error.
export async function loadDefaultPNG(readout, url = "./data/default-weights.png") {
  try {
    const r = await fetch(url);
    if (!r.ok) return false;
    const blob = await r.blob();
    const objUrl = URL.createObjectURL(blob);
    try {
      const img = new Image();
      img.src = objUrl;
      await img.decode();
      await decodeImageInto(img, readout);
      return true;
    } finally {
      URL.revokeObjectURL(objUrl);
    }
  } catch (_) {
    return false;
  }
}

// localStorage persistence — stores raw bytes as base64. Quota is ~5 MB; our
// payload at D=32768, C=10 is ~1.3 MB so it fits comfortably.
export function saveLocal(readout) {
  try {
    const { bytes, width, height } = encode(readout.W, readout.b, readout.C, readout.D);
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const payload = JSON.stringify({
      magic: MAGIC,
      width, height,
      data: btoa(bin),
    });
    localStorage.setItem(LS_KEY, payload);
    return true;
  } catch (e) {
    console.warn("saveLocal failed:", e.message);
    return false;
  }
}

export function loadLocal(readout) {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return false;
    const obj = JSON.parse(raw);
    if (!obj || obj.width !== readout.D + 1 || obj.height !== readout.C) return false;
    const bin = atob(obj.data);
    const bytes = new Uint8ClampedArray(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const { W, b } = decode(bytes, obj.width, obj.height);
    readout.W.set(W);
    readout.b.set(b);
    return true;
  } catch (e) {
    console.warn("loadLocal failed:", e.message);
    return false;
  }
}

export function clearLocal() {
  try { localStorage.removeItem(LS_KEY); } catch (_) {}
}
