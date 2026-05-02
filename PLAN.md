# ShaderNN v2.1 — fix train/test gap on live drawings + bundle pre-trained weights

## Context

Current state: the LSM reaches ~80–90% MNIST validation accuracy, but live drawings collapse to a small set of digits (typically 3, 5, 7). This is **train/test distribution shift**, not a learning bug — the readout never sees inputs that look like browser drawings.

Three concrete mismatches:

1. **Stroke appearance.** `src/draw.js` sets `imageSmoothingEnabled = false` and draws with `lineWidth: 2.0` directly into a 28×28 buffer. Strokes are hard pixel blocks. MNIST digits are anti-aliased with soft edges.
2. **Stroke width and stroke-style variance.** Trained on a single canonical width; user drawings vary widely.
3. **Reservoir extrapolation.** A chaotic recurrent system produces feature vectors highly sensitive to input statistics. Off-distribution inputs land in regions of feature space the linear readout has never seen, so it falls back to whichever classes have the largest bias term.

Two requests in this round:
- Make live drawings actually work.
- Train once, bundle the result, so the page comes up usable.

User decisions:
- Offline training mechanism: **browser-as-trainer + bundled PNG**.
- Architectural fixes: drawing-pipeline match, MNIST augmentation, raw-pixel concat. (TTA skipped.)

## Plan

### 1. Drawing pipeline match — make strokes look like MNIST

`src/draw.js`:
- Replace 28×28 internal buffer with a 140×140 (5× supersampled) buffer drawn with anti-aliasing on.
- Pen `lineWidth` becomes ~12 in buffer-pixels (≈2.4 in 28×28).
- `readBuffer()` downsamples 140→28 by area-averaging (drawImage onto a 28×28 staging canvas with high smoothing quality).

`src/preprocess.js`:
- After bbox-fit + CoM-center, apply a single 3×3 Gaussian blur (σ≈0.6).

### 2. MNIST training augmentation

New file `src/augment.js`:
- Affine: random translation ±2 px, scale 0.85–1.15, rotation ±10° via inverse-warp + bilinear sample.
- Stroke-width jitter: with prob 0.3, dilate (3×3 max) or erode (3×3 min).
- Pixel noise: add ±0.05 uniform noise, clamp [0,1].

`src/mnist.js`: `nextTrain()` returns augmented image. `getTest()` stays unaugmented.

### 3. Raw pixels concatenated to features

`src/main.js`:
- Combined feature vector length `featureDim + 784 = 33552`.
- After `sim.present()`, copy reservoir features then preprocessed image into a reusable buffer.
- `buildReadout(combinedDim, CFG.numClasses)`.

`src/persist.js`:
- PNG layout already adapts (D pulled from readout). Bump `LS_KEY` to `v2`.

### 4. Browser-as-trainer + bundled default weights

`src/persist.js`:
- New `loadDefaultPNG(readout, url)`: fetch + decode same as `loadPNG`. Silent on 404.

`src/main.js`:
- Init order: `loadLocal` → `loadDefaultPNG` → zeros.
- `?train` URL flag: disable live inference, bump `trainBatchPerFrame` for fast wall-clock training.

Workflow:
1. Open page with `?train`, wait ~20–30 min until val plateaus.
2. Click "save weights (PNG)" → downloads PNG.
3. Move to `data/default-weights.png`.
4. Every visitor now loads pre-trained weights at first paint.

## Files modified

- `src/draw.js`, `src/preprocess.js`, `src/mnist.js`, `src/main.js`, `src/persist.js`

## Files added

- `src/augment.js`

## Verification

1. `python3 -m http.server 8000` → page loads, no console errors.
2. Draw a 7 — strokes should look soft-edged, not blocky.
3. Train acc EMA above 75% within 30s; val acc above 80% within 2 min.
4. Drawing accuracy: 7/10 digits should be #1 prediction; collapse to 3/5/7 should be gone.
5. Save PNG → place at `data/default-weights.png` → clear localStorage → reload → val acc should be non-`—` after first eval cycle.
6. If `data/default-weights.png` doesn't exist, fetch fails silently.
