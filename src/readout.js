// One-hidden-layer MLP: features → ReLU(W1·x + b1) → softmax(W2·h + b2).
// Adam (with decoupled weight decay) on both layers.
//
// Trained on CPU per-example (online SGD style). For D=33552, H=64 this is
// ~2.1M params in W1 and ~640 in W2 — small enough to step at >100 ex/s in JS.

import { CFG } from "./config.js";

export function buildReadout(featureDim, numClasses, hiddenDim = CFG.hiddenDim) {
  const D = featureDim, H = hiddenDim, C = numClasses;

  // He init for ReLU layer; Glorot for output.
  const sigma1 = Math.sqrt(2 / D);
  const sigma2 = Math.sqrt(2 / (H + C));
  const W1 = new Float32Array(H * D);
  const b1 = new Float32Array(H);
  const W2 = new Float32Array(C * H);
  const b2 = new Float32Array(C);
  for (let i = 0; i < W1.length; i++) W1[i] = randn() * sigma1;
  for (let i = 0; i < W2.length; i++) W2[i] = randn() * sigma2;

  const mW1 = new Float32Array(H * D), vW1 = new Float32Array(H * D);
  const mB1 = new Float32Array(H),     vB1 = new Float32Array(H);
  const mW2 = new Float32Array(C * H), vW2 = new Float32Array(C * H);
  const mB2 = new Float32Array(C),     vB2 = new Float32Array(C);

  const z1 = new Float32Array(H);   // pre-activation
  const h  = new Float32Array(H);   // post-ReLU
  const dh = new Float32Array(H);   // hidden gradient
  const logits = new Float32Array(C);
  const probs  = new Float32Array(C);

  // Live diagnostics (read by main.js)
  const stats = { hiddenActiveRate: 0, lossEMA: 2.302, conf: 0.1 };
  let t = 0;

  function forward(features) {
    // Hidden layer: z1 = W1 · x + b1, h = max(0, z1)
    for (let j = 0; j < H; j++) {
      let z = b1[j];
      const base = j * D;
      for (let i = 0; i < D; i++) z += W1[base + i] * features[i];
      z1[j] = z;
      h[j] = z > 0 ? z : 0;
    }
    // Output layer: logits = W2 · h + b2
    for (let c = 0; c < C; c++) {
      let z = b2[c];
      const base = c * H;
      for (let j = 0; j < H; j++) z += W2[base + j] * h[j];
      logits[c] = z;
    }
    // Softmax
    let m = -Infinity;
    for (let c = 0; c < C; c++) if (logits[c] > m) m = logits[c];
    let sum = 0;
    for (let c = 0; c < C; c++) { probs[c] = Math.exp(logits[c] - m); sum += probs[c]; }
    const inv = 1 / sum;
    for (let c = 0; c < C; c++) probs[c] *= inv;
    return probs;
  }

  function predict(features) {
    forward(features);
    let bestC = 0, bestP = -1;
    for (let c = 0; c < C; c++) if (probs[c] > bestP) { bestP = probs[c]; bestC = c; }
    return { label: bestC, prob: bestP, probs };
  }

  function train(features, label) {
    forward(features);
    const lr = CFG.readoutLR;
    const wd = CFG.readoutL2;
    const b1m = CFG.adamBeta1;
    const b2m = CFG.adamBeta2;
    const eps = CFG.adamEps;

    t++;
    const bc1 = 1 - Math.pow(b1m, t);
    const bc2 = 1 - Math.pow(b2m, t);
    const stepW = lr / bc1;

    // dLogits = probs - one_hot(label)
    // dW2[c,j] = dLogits[c] * h[j]
    // dh[j]   = sum_c W2[c,j] * dLogits[c]   (gated by ReLU mask below)
    // dW1[j,i]= dh[j] * x[i] * (z1[j] > 0)
    for (let j = 0; j < H; j++) dh[j] = 0;

    for (let c = 0; c < C; c++) {
      const dz = probs[c] - (c === label ? 1 : 0);
      const base = c * H;
      // bias
      const gB = dz;
      const m_ = b1m * mB2[c] + (1 - b1m) * gB;
      const v_ = b2m * vB2[c] + (1 - b2m) * gB * gB;
      mB2[c] = m_; vB2[c] = v_;
      b2[c] -= stepW * m_ / (Math.sqrt(v_ / bc2) + eps);
      // weights
      for (let j = 0; j < H; j++) {
        const idx = base + j;
        dh[j] += W2[idx] * dz;
        const g = dz * h[j];
        const mw = b1m * mW2[idx] + (1 - b1m) * g;
        const vw = b2m * vW2[idx] + (1 - b2m) * g * g;
        mW2[idx] = mw; vW2[idx] = vw;
        W2[idx] -= stepW * mw / (Math.sqrt(vw / bc2) + eps) + lr * wd * W2[idx];
      }
    }

    // Backprop through ReLU and update W1, b1
    let activeCount = 0;
    for (let j = 0; j < H; j++) {
      if (z1[j] <= 0) { dh[j] = 0; continue; }
      activeCount++;
      const dzj = dh[j];
      // bias
      const mb = b1m * mB1[j] + (1 - b1m) * dzj;
      const vb = b2m * vB1[j] + (1 - b2m) * dzj * dzj;
      mB1[j] = mb; vB1[j] = vb;
      b1[j] -= stepW * mb / (Math.sqrt(vb / bc2) + eps);
      // weights
      const base = j * D;
      for (let i = 0; i < D; i++) {
        const idx = base + i;
        const g = dzj * features[i];
        const mw = b1m * mW1[idx] + (1 - b1m) * g;
        const vw = b2m * vW1[idx] + (1 - b2m) * g * g;
        mW1[idx] = mw; vW1[idx] = vw;
        W1[idx] -= stepW * mw / (Math.sqrt(vw / bc2) + eps) + lr * wd * W1[idx];
      }
    }

    // Diagnostics
    const pCorrect = Math.max(probs[label], 1e-12);
    const loss = -Math.log(pCorrect);
    stats.lossEMA = stats.lossEMA * 0.98 + loss * 0.02;
    stats.hiddenActiveRate = stats.hiddenActiveRate * 0.95 + (activeCount / H) * 0.05;
    let bestC = 0, bestP = -1;
    for (let c = 0; c < C; c++) if (probs[c] > bestP) { bestP = probs[c]; bestC = c; }
    stats.conf = stats.conf * 0.95 + bestP * 0.05;

    return { predicted: bestC, correct: bestC === label, probs };
  }

  function reset() {
    for (let i = 0; i < W1.length; i++) W1[i] = randn() * sigma1;
    for (let i = 0; i < W2.length; i++) W2[i] = randn() * sigma2;
    b1.fill(0); b2.fill(0);
    mW1.fill(0); vW1.fill(0); mB1.fill(0); vB1.fill(0);
    mW2.fill(0); vW2.fill(0); mB2.fill(0); vB2.fill(0);
    t = 0;
    stats.hiddenActiveRate = 0; stats.lossEMA = 2.302; stats.conf = 0.1;
  }

  return { forward, predict, train, reset, stats, W1, b1, W2, b2, D, H, C };
}

function randn() {
  const u = Math.max(1e-9, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
