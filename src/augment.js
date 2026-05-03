// Random data augmentation for MNIST training images. Helps the readout
// generalize to off-distribution inputs (e.g. user drawings) by exposing it
// to affine transforms, stroke-width changes, elastic deformation, and
// pixel noise during training.

const N = 28;
const CENTER = (N - 1) / 2;

const buf  = new Float32Array(N * N);
const tmp  = new Float32Array(N * N);
const dx   = new Float32Array(N * N);
const dy   = new Float32Array(N * N);
const dxs  = new Float32Array(N * N);
const dys  = new Float32Array(N * N);

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

// Separable 5-tap Gaussian (σ≈1.5) used to smooth the elastic displacement field.
const GK5 = [0.06136, 0.24477, 0.38774, 0.24477, 0.06136];

function blur5(src, dst, work) {
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) {
        const xx = Math.max(0, Math.min(N - 1, x + k));
        s += src[y * N + xx] * GK5[k + 2];
      }
      work[y * N + x] = s;
    }
  }
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) {
        const yy = Math.max(0, Math.min(N - 1, y + k));
        s += work[yy * N + x] * GK5[k + 2];
      }
      dst[y * N + x] = s;
    }
  }
}

function elastic(src, dst, alpha) {
  for (let i = 0; i < N * N; i++) {
    dx[i] = (Math.random() * 2 - 1);
    dy[i] = (Math.random() * 2 - 1);
  }
  blur5(dx, dxs, tmp);
  blur5(dy, dys, tmp);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      dst[i] = sample(src, x + dxs[i] * alpha, y + dys[i] * alpha);
    }
  }
}

export function augment(image) {
  const tx = (Math.random() * 2 - 1) * 3;
  const ty = (Math.random() * 2 - 1) * 3;
  const scale = 0.75 + Math.random() * 0.50;
  const rot = (Math.random() * 2 - 1) * (15 * Math.PI / 180);
  affine(image, buf, scale, rot, tx, ty);

  if (Math.random() < 0.30) {
    elastic(buf, tmp, 3);
    buf.set(tmp);
  }

  const r = Math.random();
  if (r < 0.20) {
    dilate(buf, tmp);
    buf.set(tmp);
  } else if (r < 0.30) {
    dilate(buf, tmp);
    dilate(tmp, buf);
  } else if (r < 0.45) {
    erode(buf, tmp);
    buf.set(tmp);
  }

  for (let i = 0; i < buf.length; i++) {
    const v = buf[i] + (Math.random() - 0.5) * 0.16;
    buf[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return buf;
}
