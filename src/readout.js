// Linear softmax classifier: logits = W · features + b, trained with SGD on
// cross-entropy loss with momentum and small L2 regularization.
//
// W is stored row-major: W[c * D + i] = weight from feature i to class c.

import { CFG } from "./config.js";

export function buildReadout(featureDim, numClasses) {
  const D = featureDim, C = numClasses;
  const W = new Float32Array(C * D);     // init zeros — softmax has unique optimum
  const b = new Float32Array(C);
  const vW = new Float32Array(C * D);    // momentum buffer
  const vB = new Float32Array(C);
  const logits = new Float32Array(C);
  const probs  = new Float32Array(C);

  function forward(features) {
    for (let c = 0; c < C; c++) {
      let z = b[c];
      const base = c * D;
      for (let i = 0; i < D; i++) z += W[base + i] * features[i];
      logits[c] = z;
    }
    // Softmax (numerically stable)
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

  // SGD step on a single example. label ∈ [0, C). Returns predicted label
  // and whether it was correct (for accuracy tracking).
  function train(features, label) {
    forward(features);
    const lr = CFG.readoutLR;
    const l2 = CFG.readoutL2;
    const mom = CFG.readoutMomentum;

    // gradient_c = probs[c] - one_hot[c]
    // dL/dW[c,i] = grad_c * features[i] + l2 * W[c,i]
    for (let c = 0; c < C; c++) {
      const grad = probs[c] - (c === label ? 1 : 0);
      const base = c * D;
      // momentum update on bias
      vB[c] = mom * vB[c] - lr * grad;
      b[c] += vB[c];
      // momentum update on weights
      for (let i = 0; i < D; i++) {
        const g = grad * features[i] + l2 * W[base + i];
        const v = mom * vW[base + i] - lr * g;
        vW[base + i] = v;
        W[base + i] += v;
      }
    }

    let bestC = 0, bestP = -1;
    for (let c = 0; c < C; c++) if (probs[c] > bestP) { bestP = probs[c]; bestC = c; }
    return { predicted: bestC, correct: bestC === label, probs };
  }

  function reset() {
    W.fill(0); b.fill(0); vW.fill(0); vB.fill(0);
  }

  return { forward, predict, train, reset, W, b, D, C };
}
