# ShaderNN

A neural network that runs entirely inside a graphics shader.

![demo](docs/demo.gif)
<!-- TODO: replace with actual GIF -->

---

## Why

Running a neural network normally requires a machine learning runtime — TensorFlow, PyTorch, or a server with a GPU. That overhead is fine for large models. It is overkill for the small models that increasingly need to run in browsers, games, phones, and embedded devices.

Every modern device already has a graphics card running shaders — small programs that do math in parallel. A neural network is also just math. So in principle, the graphics card can run a neural network directly, with no ML framework attached.

I wanted to find out how far that idea actually goes.

## What I Built

A working demo where a small neural network grows a shape from a single pixel inside the browser, and rebuilds itself when you erase part of it. The whole thing runs client-side: open the page and the GPU does the rest.

Two parts are unusual:

- **The model is shipped as two grayscale PNG images** (~6 KB total). The trained weights are stored as pixels. You can open the files in any image viewer.
- **The model runs as a graphics shader.** The forward pass is written in GLSL, the same language games use to draw 3D scenes — no JavaScript inference, no WASM, no backend.

I built this in Summer–Fall 2024 to teach myself shader programming and how neural networks work once you strip away the frameworks. I recently refactored it into a clean, standalone piece for this portfolio.

## How

Three pieces, kept deliberately simple:

1. **Training** — A small neural network (about 6,000 parameters, two layers) is trained in Python with PyTorch. Training is *quantization-aware*, meaning the network learns weights that still work after being compressed to 8 bits.
2. **Export** — The trained weights are saved as two PNG files, one per layer.
3. **Inference** — The browser loads the PNGs, hands the values to the GPU, and a single fragment shader runs the full network for 65,536 cells in parallel, every frame, at 60 fps.

The model itself is intentionally small. The interesting part is the delivery: a static page, two images, and a shader.

## Available Models

Each target shape is a separately trained network. The trained weights live in `data/` as image files — open them and you are looking at the model.

| Target | fc1 weights | fc2 weights | Metadata |
|---|---|---|---|
| Heart | `data/ca_fc1_heart.png` | `data/ca_fc2_heart.png` | `data/ca_weights_heart.json` |
| Smiley *(placeholder)* | `data/ca_fc1_smiley.png` | `data/ca_fc2_smiley.png` | `data/ca_weights_smiley.json` |
| Lizard *(placeholder)* | `data/ca_fc1_lizard.png` | `data/ca_fc2_lizard.png` | `data/ca_weights_lizard.json` |

Each pair of PNGs is about 6 KB. The JSON file holds the per-layer biases and the dequantization scale/min used to map the 8-bit pixel values back to floats.

## What If

If small models can ship as images and run inside graphics shaders, the deployment story changes:

- **Any static website** can host a working ML model. No backend, no API, no framework.
- **Games and creative tools** can use ML directly inside their existing rendering pipeline.
- **Image hosting becomes model hosting.** Any CDN or GitHub repo becomes a model registry.
- **Edge devices** that have a GPU but no ML runtime can still run inference using graphics code they already support.

This is a portfolio project, not a product. But the underlying question — *what is the smallest, most portable way to ship a working neural network?* — is one I expect to keep mattering.

## Run Locally

```bash
git clone <repo-url>
cd ShaderNN

python -m venv .venv && source .venv/bin/activate
pip install torch numpy Pillow

python build_ca.py --target heart   # trains the network, exports two PNGs
python -m http.server 8765          # serves the demo
```

Open `http://localhost:8765`.
