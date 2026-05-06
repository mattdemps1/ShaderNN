import { CFG } from "./config.js";

const N = CFG.numChannels;   // 16
const H = CFG.hiddenDim;     // 96
const P = CFG.perceptionDim; // 48

// Decode a grayscale PNG to a Float32Array via OffscreenCanvas.
// Returns raw [0,1] values from the R channel (rows*cols elements).
async function decodePng(url, rows, cols) {
  const blob   = await fetch(url).then(r => {
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.blob();
  });
  const bitmap = await createImageBitmap(blob);
  const oc  = new OffscreenCanvas(cols, rows);
  const ctx = oc.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const raw = ctx.getImageData(0, 0, cols, rows).data; // Uint8ClampedArray RGBA
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) out[i] = raw[i * 4] / 255; // R channel
  return out;
}

// Pack a Float32Array into a Float32Array aligned to vec4 slots.
// std140 float[] arrays require 16-byte stride; using vec4[] avoids that.
// Each group of 4 floats occupies one vec4 slot (4 × 4 = 16 bytes).
function packVec4(floats) {
  const slots = Math.ceil(floats.length / 4);
  const out   = new Float32Array(slots * 4);
  out.set(floats);
  return out;
}

// Load weight PNGs for a given target, dequantize, and upload the Weights UBO.
// Returns { ubo } — caller binds it via ca.bindWeights(ubo).
export async function loadWeights(gl, target = "heart") {
  const dir = CFG.dataDir;

  const meta = await fetch(`${dir}/ca_weights_${target}.json`).then(r => {
    if (!r.ok) throw new Error(`ca_weights_${target}.json: ${r.status}`);
    return r.json();
  });

  // fc1: shape (H=96, P=48) → PNG 48 cols × 96 rows
  const fc1Raw = await decodePng(`${dir}/ca_fc1_${target}.png`, H, P);
  // fc2: shape (N=16, H=96) → PNG 96 cols × 16 rows
  const fc2Raw = await decodePng(`${dir}/ca_fc2_${target}.png`, N, H);

  const fc1 = new Float32Array(H * P);
  const fc2 = new Float32Array(N * H);
  for (let i = 0; i < H * P; i++) fc1[i] = fc1Raw[i] * meta.fc1_scale + meta.fc1_min;
  for (let i = 0; i < N * H; i++) fc2[i] = fc2Raw[i] * meta.fc2_scale + meta.fc2_min;

  const b1 = new Float32Array(meta.fc1_bias);
  const b2 = new Float32Array(meta.fc2_bias);

  // UBO layout (std140, vec4 arrays):
  //   vec4 fc1[1152]  → 18432 bytes
  //   vec4 b1[24]     →   384 bytes
  //   vec4 fc2[384]   →  6144 bytes
  //   vec4 b2[4]      →    64 bytes
  //                  = 25024 bytes total
  const uboData = new Float32Array((1152 + 24 + 384 + 4) * 4);
  let off = 0;
  uboData.set(packVec4(fc1), off); off += 1152 * 4;
  uboData.set(packVec4(b1),  off); off += 24   * 4;
  uboData.set(packVec4(fc2), off); off += 384  * 4;
  uboData.set(packVec4(b2),  off);

  const maxUBO = gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE);
  if (maxUBO < uboData.byteLength) {
    throw new Error(`UBO too small: need ${uboData.byteLength} B, max is ${maxUBO} B`);
  }

  const ubo = gl.createBuffer();
  gl.bindBuffer(gl.UNIFORM_BUFFER, ubo);
  gl.bufferData(gl.UNIFORM_BUFFER, uboData, gl.STATIC_DRAW);
  gl.bindBuffer(gl.UNIFORM_BUFFER, null);

  console.log(`[weights] fc1 ∈ [${meta.fc1_min.toFixed(3)}, ${(meta.fc1_min + meta.fc1_scale).toFixed(3)}]`);
  console.log(`[weights] fc2 ∈ [${meta.fc2_min.toFixed(3)}, ${(meta.fc2_min + meta.fc2_scale).toFixed(3)}]`);
  console.log(`[weights] UBO ${uboData.byteLength} B / max ${maxUBO} B`);

  return { ubo };
}
