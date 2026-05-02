// MNIST-style preprocessing for hand-drawn 28×28 digits.
//
// MNIST original pipeline:
//   1. find bounding box of foreground pixels
//   2. resize bbox to fit in a 20×20 box, preserving aspect
//   3. paste into a 28×28 canvas, centered by center-of-mass
//
// We follow the same recipe so live drawings match the training distribution.

const N = 28;
const FIT = 20;
const THRESH = 0.05;

const out = new Float32Array(N * N);
const tmp = new Float32Array(FIT * FIT);
const blurBuf = new Float32Array(N * N);

// 3×3 Gaussian (σ≈0.6) — matches the soft edges MNIST has from its
// 20×20 → 28×28 anti-aliased rescaling pipeline.
const GK = [
  1/16, 2/16, 1/16,
  2/16, 4/16, 2/16,
  1/16, 2/16, 1/16,
];

function gaussBlur(src) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= N) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= N) continue;
          s += src[yy * N + xx] * GK[(dy + 1) * 3 + (dx + 1)];
        }
      }
      blurBuf[y * N + x] = s;
    }
  }
  src.set(blurBuf);
}

// Bilinear sample of `src` (size sw×sh, row-major) at fractional (sx, sy).
function sample(src, sw, sh, sx, sy) {
  if (sx < 0 || sy < 0 || sx > sw - 1 || sy > sh - 1) return 0;
  const x0 = Math.floor(sx), y0 = Math.floor(sy);
  const x1 = Math.min(x0 + 1, sw - 1);
  const y1 = Math.min(y0 + 1, sh - 1);
  const fx = sx - x0, fy = sy - y0;
  const a = src[y0 * sw + x0], b = src[y0 * sw + x1];
  const c = src[y1 * sw + x0], d = src[y1 * sw + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

export function preprocess(image) {
  // image: Float32Array(784), row-major top-left origin, [0,1]
  out.fill(0);

  // 1. Find bounding box of pixels above threshold
  let minX = N, minY = N, maxX = -1, maxY = -1;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (image[y * N + x] > THRESH) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return out;  // empty input

  const bbW = maxX - minX + 1;
  const bbH = maxY - minY + 1;

  // 2. Resize the bbox to fit in FIT×FIT, preserving aspect
  const scale = FIT / Math.max(bbW, bbH);
  const newW = Math.max(1, Math.round(bbW * scale));
  const newH = Math.max(1, Math.round(bbH * scale));
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const sx = minX + (x + 0.5) / scale - 0.5;
      const sy = minY + (y + 0.5) / scale - 0.5;
      tmp[y * FIT + x] = sample(image, N, N, sx, sy);
    }
  }

  // 3. Compute center-of-mass of the resized image and paste centered.
  let totalMass = 0, cx = 0, cy = 0;
  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const v = tmp[y * FIT + x];
      totalMass += v;
      cx += v * x;
      cy += v * y;
    }
  }
  cx = totalMass > 0 ? cx / totalMass : (newW - 1) / 2;
  cy = totalMass > 0 ? cy / totalMass : (newH - 1) / 2;

  // We want CoM at (N/2 - 0.5, N/2 - 0.5)
  const offX = Math.round(N / 2 - 0.5 - cx);
  const offY = Math.round(N / 2 - 0.5 - cy);

  for (let y = 0; y < newH; y++) {
    const dy = y + offY;
    if (dy < 0 || dy >= N) continue;
    for (let x = 0; x < newW; x++) {
      const dx = x + offX;
      if (dx < 0 || dx >= N) continue;
      out[dy * N + dx] = tmp[y * FIT + x];
    }
  }

  gaussBlur(out);
  return out;
}
