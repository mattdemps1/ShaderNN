# ShaderNN → Neural Cellular Automata Overhaul

## Context
MNIST failed (reservoir + static images = temporal mismatch, plateaued at 87%). Audio keyword spotting pivoted away from the project's visual core. The project's actual hooks — WebGL shaders, PNG-as-weights — need a demo where both shine. Neural Cellular Automata is the ideal fit: the model is tiny (~6K params), fits in two small PNGs, the WebGL simulation IS the demo (not just a visualization), training is offline Python, and the "self-healing living grid" is immediately compelling to anyone.

## What We're Building
A tiny neural network (~6,256 params) defines a local update rule for a 256×256 grid. Each pixel reads its 3×3 neighborhood, runs through the net, updates its own state. The grid "grows" a target image from a single seed pixel, and heals if you paint damage on it. Weights stored as two grayscale PNGs (quantized to uint8), loaded by the browser, dequantized, and run in WebGL2 shaders at 60fps. No training in browser — Python only.

---

## Architecture

### Neural CA Model
- **N=16 channels** per cell, **hidden_dim=96**
- **Perception** (fixed, not learned): for each cell, concatenate [identity, Sobel_x, Sobel_y] across all N channels → 3N=48 values
- **MLP**: fc1(48→96, ReLU) + fc2(96→16, no activation)
- **Total params**: 48×96 + 96 + 96×16 + 16 = **6,256**
- **Stochastic update**: each cell updates with 50% probability (prevents lock-step)
- **Alive mask**: cell alive if max alpha in 3×3 neighborhood > 0.1; dead cells can't update

### PNG-as-Weights (8-bit quantization)
- `data/ca_fc1.png`: 48×96 grayscale PNG (fc1 weights, shape 96×48 flattened row-major)
- `data/ca_fc2.png`: 96×16 grayscale PNG (fc2 weights, shape 16×96 flattened row-major)
- `data/ca_weights.json`: `{fc1_min, fc1_scale, fc1_bias[96], fc2_min, fc2_scale, fc2_bias[16]}`
- **Dequantization**: `w_float = uint8_val / 255.0 * scale + min`
- Biases in JSON (only 112 values, not worth a PNG)
- PNG saved as mode `'L'` (grayscale) to avoid premultiplied-alpha corruption

### WebGL2 Inference
- **State**: 4 RGBA32F textures (N=16 channels = 4 tex × 4 channels), ping-ponged
- **MRT**: writes all 4 textures in one fragment shader pass (`gl.drawBuffers`)
- **Weights**: dequantized Float32Array → WebGL2 Uniform Buffer Object (UBO)
  - Packed as `vec4` arrays (not `float[]`) to avoid std140 padding to 16 bytes/float
  - Total UBO: 25,024 bytes — well under 64KB limit on desktop
- **Damage**: circle center+radius passed as uniforms; handled inside the CA shader (cells inside circle zeroed and not updated)

---

## Files

### Create
| File | Purpose |
|------|---------|
| `build_ca.py` | Python training: QAT, 50K steps, export PNGs + JSON |
| `src/ca.js` | WebGL2 CA manager: texture alloc, MRT FBO, step/display/seed/damage |
| `src/weights.js` | PNG → uint8 → dequantize → UBO upload |

### Modify (rewrite in place)
| File | Change |
|------|--------|
| `index.html` | New layout: large CA canvas center, sidebar with weight PNGs + controls |
| `src/main.js` | New main loop: CA step, interaction (click damage, R reset, Space pause) |
| `src/config.js` | Replace with CA params: gridW/H, numChannels, hiddenDim, stepsPerFrame, etc. |
| `src/draw.js` | Deleted (MNIST-specific) |

### Keep unchanged
`src/glutil.js` — `makeFloatTex`, `makeFBO`, `bindTexUnits`, `program`, `quad`

---

## Implementation Order

1. Delete old files, skeleton `index.html` + `config.js`
2. `src/weights.js` — load PNGs, dequantize, log weight range to console (no GPU yet)
3. `src/ca.js` — ping-pong textures, CA shader, MRT
4. Wire UBO from step 2 into step 3
5. `src/main.js` — interaction: Space, R, click damage
6. Polish `index.html` sidebar, weight PNG display, HUD
7. `build_ca.py` — train heart target, export PNGs/JSON
8. End-to-end test: load PNGs in browser, verify growth + heal behavior
