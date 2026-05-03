# ShaderNN

Real-time spiking neural network in the browser. 16,384 leaky integrate-and-fire neurons run on the GPU (WebGL2 fragment shaders). A small MLP readout runs on the CPU and is trained online to classify handwritten digits from MNIST. Draw digits in the canvas to run live inference.

---

## What the colors mean

Press **W** to cycle through three visualization modes.

### Mode 0 — Membrane voltage (default)

The brain view blends two signals:

| Appearance | What it means |
|---|---|
| **Black / deep purple** | Neuron is resting — voltage low, not recently active |
| **Purple → red → orange** | Membrane voltage building up; trace accumulating from recent spikes |
| **Bright white flash** | Neuron fired *right now* — spike in this simulation step |
| **Blue-ish tint** | Raw membrane voltage mixed in |

The dominant color is a **magma colormap** (black → purple → red → orange → white) driven by the neuron's spike trace. When input arrives, a wave of color spreads through the reservoir as activity propagates through recurrent connections.

### Mode 1 — Firing rate (trace)

Pure magma colormap on the trace value that the readout actually reads as its input features:

| Color | Firing rate |
|---|---|
| Black | Silent — never fires |
| Purple | Rarely fires |
| Red / orange | Moderate activity |
| Yellow / white | High activity |

This is exactly what information the MLP readout has to work with. Neurons that stay black on every digit are not contributing to classification.

### Mode 2 — Input drive

**Viridis colormap** (dark blue → teal → green → yellow) showing how strongly each neuron is being pushed by the current image:

| Color | Drive |
|---|---|
| Dark blue | No input drive from this image |
| Teal / green | Moderate drive — fires probabilistically |
| Yellow | Strong drive — fires almost every step |

This visualizes the fixed random input projection: each neuron samples 8 random pixels from the 28×28 input with ±1 sign weights.

---

## The "Seen by Net" preview

The small box next to the prediction bars shows the **28×28 image the network actually classified**, after preprocessing (bounding-box crop → 20×20 fit → center-of-mass centering → Gaussian blur). The most important diagnostic: if your drawing looks like a recognizable digit in the preview but the prediction is still wrong, the readout hasn't generalized yet — train longer. If the preview looks like noise or a smear, the preprocessing is distorting your stroke.

---

## Parameters

### LIF neuron dynamics

| Parameter | Default | What it does |
|---|---|---|
| **vThr** (threshold) | 0.55 | Voltage a neuron must reach before firing. Higher = harder to fire = lower population rate. Lower = easier to fire = more spikes. Tune so pop. rate HUD reads 5–25%. |
| **vReset** | 0.0 | Voltage snapped to immediately after a spike. Fixed. |
| **leak** | 0.92 | Fraction of voltage kept each step. 0.92 = neuron loses 8% per step. Lower = leakier = faster decay. |
| **tRef** | 3 steps | Refractory period — neuron locked out for 3 steps after firing. Prevents runaway cascades. |
| **thrJitter** | 0.10 | Per-neuron random threshold offset, assigned at init. Desynchronizes the grid so neurons don't all fire at once. |
| **v noise** | 0.02 | Random voltage kick each step. Keeps the reservoir slightly stochastic. Set to 0 by **silent mode**. |
| **trace decay** | 0.85 | Low-pass filter on spikes: `trace = trace × decay + spike × (1 − decay)`. Half-life ≈ 4 steps. Lower = trace reacts faster but noisier. Higher = smoother but lags behind input. |

### Reservoir connectivity (fixed at startup, never trained)

| Parameter | Default | What it does |
|---|---|---|
| **resSigmaLocal** | 0.18 | Std-dev of local 3×3 stride-1 recurrent weights. Each neuron gets input from its 8 immediate neighbours. |
| **resSigmaLong** | 0.12 | Std-dev of long-range 3×3 stride-16 weights. Connects neurons 16 pixels apart — activity can reach across the full 128-row grid in ~8 hops instead of 120. |
| **resSparsity** | 0.55 | 55% of weight slots zeroed out. Sparse connectivity avoids over-coupling. |
| **longStride** | 16 | How far the long-range connections jump vertically. |

### Input projection (fixed at startup)

| Parameter | Default | What it does |
|---|---|---|
| **inputFanIn** | 8 | Each neuron randomly samples 8 pixels from the 28×28 input with ±1 sign weights. |
| **drive scale** | 2.0 | Multiplies the Poisson spike probability from input drive. Higher = stronger signal. If neurons fire on blank inputs, lower this or raise vThr. |
| **noise rate** | 0.001 | Background drizzle probability each step. Keeps the reservoir alive between inputs. Set to 0 by **silent mode**. |

### MLP readout (trained during the session)

Input features → 64 ReLU hidden units → 10 softmax outputs. Features: 32,768 spike traces + 784 raw pixels, each block L2-normalized independently.

| Parameter | Default | What it does |
|---|---|---|
| **learning rate** | 0.001 | Adam step size. Lower = slower but more stable. If the loss curve oscillates, halve it. |
| **readoutL2** | 1e-4 | AdamW weight decay. Prevents weights from growing unboundedly. |
| **hiddenDim** | 64 | Number of ReLU hidden units. More = more capacity, slower steps. |
| **pixel scale** | 0.5 | How much the raw-pixel block contributes relative to the reservoir block. 0 = reservoir only. 1 = equal. 2 = pixels dominate. |

### Training schedule

| Parameter | Default | What it does |
|---|---|---|
| **batch / frame** | 4 | MNIST examples processed per animation frame (~120 ex/s at 30 fps). More = faster training, jankier animation. |
| **valEvery** | 50 | Run a validation check every N examples. The val-acc curve updates at this cadence. |
| **valBatch** | 64 | Random test-set examples sampled for each validation check. |

---

## Controls

### Keyboard

| Key | Action |
|---|---|
| **Space** | Pause / resume training |
| **C** | Clear the drawing canvas |
| **R** | Full reset — re-rolls reservoir weights, clears readout, wipes history |
| **L** | Freeze / unfreeze readout weights. Training stops; live inference keeps running. |
| **W** | Cycle brain view: voltage → trace → input drive |

### Buttons

| Button | Action |
|---|---|
| **save (.bin)** | Download readout weights as a binary file |
| **load (.bin)** | Load a previously saved `.bin` file back into the readout |
| **reset readout** | Re-initialize readout weights to random, clear training history |
| **re-roll reservoir** | Regenerate reservoir random weights without touching the readout |

### Checkboxes

| Toggle | What it does |
|---|---|
| **silent (fire only on input)** | Sets noise rate and v noise to 0. Neurons stay dark unless driven by input. Useful for watching the reservoir respond cleanly to your drawings. |
| **augment** | Apply random affine transforms, elastic deformation, and stroke-width jitter to training images. Keep this on — it's what teaches the readout to handle imperfect live drawings. |
| **pause** | Same as Space. Stops training; inference stays live. |
| **freeze readout** | Same as L. Stops the readout from updating its weights. |

---

## Diagnostics (right column)

### Reservoir

| Readout | Healthy range | What it tells you |
|---|---|---|
| **pop. firing rate** | 5–25% | Fraction of neurons spiking in the last sampled step. Too low: can't fire — raise drive scale or lower vThr. Too high: everything fires — raise vThr. Auto-calibrated at startup. |
| **active neurons** | 800–4000 | Raw count from pop. rate × 16,384. |
| **trace mean** | 0.01–0.10 | Average spike trace across all neurons. Near zero means the reservoir isn't responding — bad features, nothing to classify on. |
| **trace std** | > 1e-3 | How much trace varies neuron-to-neuron. Near zero = all neurons look the same = no discriminative information. |
| **pixel mean** | 0.05–0.30 | Mean pixel intensity of the current input (after preprocessing). Sanity check that your drawing is registering. |

### Readout

| Readout | Healthy range | What it tells you |
|---|---|---|
| **top-1 conf EMA** | Rising toward 80%+ | How confident the readout is in its top class. Should climb as it learns. Stuck at 10% = not learning at all. |
| **hidden active rate** | 30–70% | Fraction of 64 hidden units that fired on the last example. Near 0% = dying ReLUs (lower lr). Near 100% = saturated (lower lr or pixel scale). |
| **readout dim** | 33552→64→10 | Input features → hidden units → output classes. |

### Curves

| Curve | What to look for |
|---|---|
| **Val accuracy** | Should rise from ~10% to 90%+ within 5–10 minutes. Sawtooth pattern = lr too high. |
| **Cross-entropy loss** | Falls from ~2.3 (random 10-way) toward ~0.3 as the readout learns. Stuck above 1.0 usually means weak reservoir features — check trace std and pop. rate. |
| **Population rate** | Should stay roughly flat in the 5–25% band throughout training. |

---

## Workflow: train, save, ship

1. Open the page. Reservoir auto-calibrates vThr and logs pop. rate to the DevTools console.
2. Training starts automatically at ~120 examples/second.
3. Watch loss fall and val acc rise. Target: **≥ 90% val acc** within ~5 minutes.
4. Once satisfied, click **save (.bin)**.
5. Move the file to `data/default-weights.bin`.
6. Every visitor now loads your pre-trained weights on first paint — no training required.

To force a fresh start: click **reset readout**, or open DevTools → Application → IndexedDB → delete the `shadernn` database entry.

---

## Architecture

A 128×128 grid of leaky integrate-and-fire neurons lives in two WebGL2 textures (ping-ponged). Each example: the MNIST image is projected through a fixed sparse random matrix (784 → 16,384) and Poisson-sampled into spike probabilities. A fragment shader steps all neurons in parallel — integrate input, apply recurrent weights (local 3×3 stride-1 and long-range 3×3 stride-16), leak, fire if above threshold, reset, update trace. After 24 simulation steps the trace texture is read back twice (mid snapshot at step 12, end at step 24), producing 32,768 features. Those are L2-normalized, concatenated with the L2-normalized raw pixels, and fed into a two-layer MLP (33,552 → 64 ReLU → 10 softmax) trained online with AdamW. The reservoir weights never change — only the MLP readout learns.
