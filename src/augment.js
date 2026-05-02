// Random data augmentation for MNIST training images. Helps the readout
// generalize to off-distribution inputs (e.g. user drawings) by exposing it
// to small affine transforms, stroke-width changes, and pixel noise during
// training.

const N = 28;
const CENTER = (N - 1) / 2;

const buf = new Float32Array(N * N);
const tmp = new Float32Array(N * N);

function sample(src, x, y) {
  if (x < 0 || y < 0 || x > N - 1 || y > N - 1) return 0;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, N - 1);
  const y1 = Math.min(y0 + 1, N - 1);
  const fx = x - x0, fy = y - y0;
  const a = src[y0 * N + x0], b = src[y0 * N + x1];
  const c = src[y1 * N + x0], d = src[y1 * N + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

function affine(src, dst, scale, rot, tx, ty) {
  const cos = Math.cos(rot) / scale;
  const sin = Math.sin(rot) / scale;
  for (let y = 0; y < N; y++) {
    const dy = y - CENTER - ty;
    for (let x = 0; x < N; x++) {
      const dx = x - CENTER - tx;
      const sx =  cos * dx + sin * dy + CENTER;
      const sy = -sin * dx + cos * dy + CENTER;
      dst[y * N + x] = sample(src, sx, sy);
    }
  }
}

function dilate(src, dst) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let m = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= N) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= N) continue;
          const v = src[yy * N + xx];
          if (v > m) m = v;
        }
      }
      dst[y * N + x] = m;
    }
  }
}

function erode(src, dst) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let m = 1;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= N) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= N) continue;
          const v = src[yy * N + xx];
          if (v < m) m = v;
        }
      }
      dst[y * N + x] = m;
    }
  }
}

export function augment(image) {
  const tx = (Math.random() * 2 - 1) * 2;
  const ty = (Math.random() * 2 - 1) * 2;
  const scale = 0.85 + Math.random() * 0.30;
  const rot = (Math.random() * 2 - 1) * (10 * Math.PI / 180);
  affine(image, buf, scale, rot, tx, ty);

  const r = Math.random();
  if (r < 0.15) {
    dilate(buf, tmp);
    buf.set(tmp);
  } else if (r < 0.30) {
    erode(buf, tmp);
    buf.set(tmp);
  }

  for (let i = 0; i < buf.length; i++) {
    const v = buf[i] + (Math.random() - 0.5) * 0.10;
    buf[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return buf;
}
