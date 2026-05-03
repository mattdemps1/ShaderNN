# ShaderNN v3 — make the spiking reservoir actually classify MNIST + live drawings

## Context

The current build hits all four failure modes the user reported: val accuracy plateaus low (<70%), live drawings collapse to a few digits, training oscillates, and what little progress happens is glacial. The user wants to keep the spiking-reservoir architecture (one LIF per pixel on the GPU) as the core of the project, so the fix is to repair the LSM, not replace it.

Diagnosis from reading `src/sim.js`, `src/main.js`, `src/readout.js`, `src/config.js`, `shaders/neuron.frag.glsl`, `shaders/input.frag.glsl`:

### Why the readout doesn't learn well
- **Optimizer is unstable.** `readoutLR: 0.5` + `readoutMomentum: 0.9` + `focalGamma: 2.0` is an aggressive combination on a single-example online SGD. Focal loss is for class imbalance (not present in MNIST) and starves learning once `pCorrect` rises (γ=2 → focal=0.01 at p=0.9). Combined with momentum=0.9 and a fat 33,552-dim feature vector, updates oscillate.
- **Feature blocks are not commensurable.** `featurize()` in `src/main.js:53–65` concatenates 32,768 reservoir-trace values (typical magnitude ~0.05) with 784 raw pixels (typical magnitude ~0.5), then L2-normalizes the *combined* vector. After normalization the raw-pixel block carries most of the L2 mass per element and dominates the gradient. The reservoir contributes proportionally little signal — so we're effectively training a noisy linear-on-pixels classifier, which caps near 85%.
- **Reservoir features are weak.** `traceDecay=0.94` over `presentSteps=48` means by step 48 the trace is `rate * (1 - 0.94^48) ≈ 0.95 * rate` — okay for the end snapshot but the mid snapshot at step 24 is only `0.78 * rate`. With `inputDriveScale=1.4` the population rate is also borderline; if it sits below ~3% the trace block is mostly zeros for any given example.

### Why training is slow
- **Synchronous GPU readback.** `sim.present()` calls `gl.readPixels` twice per example (mid + end snapshot), and `populationRate()` does a third every frame. `readPixels` is a hard pipeline stall. At `trainBatchPerFrame=1` we get ~30 examples/sec → 1 epoch over 60k MNIST takes >30 min.
- **48 sim steps per example.** With trace already saturating after ~30 steps we're paying for steady-state we don't need.

### Why live drawings collapse
- Once the val classifier is biased toward whichever pixels happen to dominate the linear weights, off-distribution drawings hit a small region of feature-space the readout has been trained on and snap to the same handful of classes. Augmentation (`src/augment.js`) is moderate (±2px translate, ±10° rotation, scale 0.85–1.15) and does not cover the variability of human strokes (slant, pen pressure, sloppy connections).

## Plan

Five focused changes. They compose; none of them require restructuring the reservoir/shader pipeline.

### 1. Fix the readout optimizer (`src/readout.js`, `src/config.js`)

Replace SGD-with-momentum + focal loss with **Adam** + plain cross-entropy. Adam adapts step sizes per-parameter, which is what an L2-normalized but very long feature vector needs.

- New CFG: `readoutOptimizer: 'adam'`, `readoutLR: 1e-3`, `adamBeta1: 0.9`, `adamBeta2: 0.999`, `adamEps: 1e-8`, `readoutL2: 1e-4` (small weight decay, applied as decoupled AdamW-style), `focalGamma: 0` (off).
- Add `mW`, `vW`, `mB`, `vB` Float32Arrays in `buildReadout`. Replace the SGD update block in `train()`:
  ```
  m = β1·m + (1-β1)·g
  v = β2·v + (1-β2)·g²
  W -= lr · m̂ / (√v̂ + ε) + lr · l2 · W
  ```
  Track `t` as a per-readout step counter for bias correction.
- Keep `forward`, `predict`, PNG persistence shape unchanged.

### 2. Fix feature-block balance (`src/main.js`)

Normalize the reservoir block and the pixel block **independently**, then concatenate. Both contribute equal L2 mass and the linear readout learns from both.

In `featurize()` at `src/main.js:53–65`:
- L2-normalize `f` (length 32,768) in place.
- L2-normalize `image28` (length 784) into the tail of `combinedFeatures`.
- Drop the combined-norm pass.
- Optional small scaling: multiply pixel block by `0.5` so reservoir keeps slight edge — the reservoir is the whole point of the project.

### 3. Tune the reservoir for fast, informative trace (`src/config.js`, `src/sim.js`)

Make the trace build faster and ensure population rate sits in 5–20%.

- `traceDecay: 0.94 → 0.85`. Trace half-life drops from ~11 steps to ~4 steps.
- `presentSteps: 48 → 24`, `midSnapshotStep: 24 → 12`. Halves GPU work per example.
- `inputDriveScale: 1.4 → 2.0`, `noiseRate: 0.002 → 0.001`. Ensures clean signal-driven activity.
- After init, log first-frame `populationRate()` to console; if outside [0.05, 0.25] auto-bump `vThr` ±0.05 and re-check (one-shot calibration in `main.js`, not a runtime loop).

### 4. Faster training loop (`src/main.js`, `src/sim.js`)

Two cheap wins; no PBO/async-readback rewrite needed.

- Remove the per-frame `populationRate()` readback. Move it behind `viewMode` change or a 1-Hz timer (`if (t - lastPopRead > 1000)`); keep the HUD updating from a cached value.
- `trainBatchPerFrame: 1 → 4` (and `?train` mode keeps its `6`). With presentSteps halved and pop-rate readback gone, we should comfortably do 4 examples/frame.
- Combined effect: ~120 examples/sec instead of ~30. One epoch in ~8 min instead of >30.

### 5. Stronger MNIST augmentation (`src/augment.js`)

Cover the human-drawing distribution.

- Rotation `±10° → ±15°`. Translation `±2px → ±3px`. Scale `0.85–1.15 → 0.75–1.25`.
- Dilate/erode probability `0.30 → 0.45`, and add a "double dilate" path (apply 3×3 max twice, ~5px effective stroke) so the readout sees fat strokes too.
- Pixel noise `±0.05 → ±0.08`.
- Add **elastic deformation** (small random displacement field, σ=4, scale=2) on 30% of examples — the standard MNIST augmentation. Implement as: random Gaussian-smoothed displacement field, bilinear sample. ~30 lines.

## Files modified

- `src/config.js` — new optimizer/reservoir/training params
- `src/readout.js` — Adam optimizer in place of SGD-momentum + focal
- `src/main.js` — split-block normalization in `featurize`, throttle `populationRate`, calibrate `vThr` once at init
- `src/sim.js` — add a "skip pop-rate" code path, no behavioral change required if `main.js` just calls less often
- `src/augment.js` — wider ranges + elastic deformation

No shader changes. No new files.

## Verification

Run `python3 -m http.server 8000` and open the page. Acceptance bar:

1. **Console at startup**: `populationRate` prints in [0.05, 0.25]. If not, `vThr` auto-calibration message visible.
2. **Speed**: HUD `examples seen` increments by ~120/sec (visible by watching for 5 sec).
3. **Stability**: train acc EMA monotonically rises — no drop-back-to-10% after early peak. Val acc curve smooth (no sawtooth >5%).
4. **Accuracy**: val acc ≥ 90% within 5 min of training. ≥ 93% within 15 min.
5. **Live drawings**: with val ≥ 90%, draw 0–9 each twice → at least 16/20 correct top-1. No more than 2 of those 20 should map to the same wrong class (i.e., no "everything is a 3" collapse).
6. **Save → reload**: click save weights → place at `data/default-weights.png` → clear localStorage → reload → val acc on first eval cycle ≥ 90%.

If any of (3), (4), (5) fail, the most likely follow-up is to push the reservoir block harder — increase `presentSteps` back to 36, or add a third snapshot at step 18 (3× features).
