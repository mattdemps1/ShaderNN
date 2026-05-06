"""
Neural Cellular Automata trainer.
Trains a tiny MLP (~6,256 params) to grow and maintain a target image.
Exports weights as quantized uint8 PNG files for the ShaderNN browser demo.

Usage:
    python build_ca.py [--target heart|smiley|lizard] [--steps N] [--no-cuda]

Dependencies:
    pip install torch numpy Pillow
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Hyperparameters ────────────────────────────────────────────────────────────
N            = 16       # channels per cell
H_DIM        = 96       # hidden dimension
GRID         = 256      # inference grid (browser); training uses GRID_TRAIN
GRID_TRAIN   = 64       # small training grid: 4× fewer cells → longer rollouts fit in VRAM
TARGET_SIZE  = 40       # target image is centred in a 40×40 region
BATCH        = 4
POOL_SIZE    = 256
LR           = 1e-3
TRAIN_STEPS  = 30_000
CA_STEPS_MIN = 16       # longer rollouts: gradient reaches the boundary region
CA_STEPS_MAX = 24       # fits in VRAM without checkpointing at GRID=64
DAMAGE_EVERY = 8
DAMAGE_HALF  = BATCH // 2
DAMAGE_R     = 6        # scaled down for 64×64 grid
ALIVE_THRESH = 0.1

# Sobel kernels applied as depthwise conv2d (groups=N)
KX = torch.tensor([[-1., 0., 1.],
                    [-2., 0., 2.],
                    [-1., 0., 1.]]) / 8.0
KY = KX.T


# ── Quantization-aware helpers ─────────────────────────────────────────────────

def fake_quantize(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: quantize in forward, real grads in backward."""
    x_min = x.detach().min()
    x_max = x.detach().max()
    scale = (x_max - x_min).clamp(min=1e-8)
    q = ((x - x_min) / scale * 255.0).round().clamp(0, 255)
    # Dequantize
    x_q = q / 255.0 * scale + x_min
    # Straight-through: forward value is quantized, gradients flow as if identity
    return x_q.detach() + (x - x.detach())


def quantize_to_uint8(w: torch.Tensor):
    """Final export quantization. Returns (uint8_np, w_min, w_scale)."""
    w_min   = w.min().item()
    w_max   = w.max().item()
    scale   = w_max - w_min
    uint8   = np.round((w.detach().cpu().numpy() - w_min) / max(scale, 1e-8) * 255)
    uint8   = uint8.clip(0, 255).astype(np.uint8)
    return uint8, w_min, scale


# ── Target image generators ────────────────────────────────────────────────────

def _blank(size=TARGET_SIZE):
    return np.zeros((size, size, 4), dtype=np.float32)


def draw_heart(size=TARGET_SIZE):
    img = _blank(size)
    cx, cy = size / 2.0, size / 2.0
    scale = size * 0.38  # ~15 px radius for size=40
    for y in range(size):
        for x in range(size):
            # Algebraic heart curve: (x²+y²-1)³ - x²y³ ≤ 0
            # No y-flip: WebGL displays texture row 0 at canvas bottom, so we
            # want lobes (math y>0) at numpy row size-1 to land at canvas top.
            nx = (x - cx) / scale
            ny = (y - cy) / scale
            if (nx**2 + ny**2 - 1)**3 - nx**2 * ny**3 <= 0:
                img[y, x] = [0.87, 0.18, 0.18, 1.0]
    return img


def draw_smiley(size=TARGET_SIZE):
    img = _blank(size)
    cx, cy, r = size / 2, size / 2, size * 0.42
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            in_face = dist < r
            in_eye_l = math.sqrt((dx + r*0.32)**2 + (dy + r*0.20)**2) < r * 0.13
            in_eye_r = math.sqrt((dx - r*0.32)**2 + (dy + r*0.20)**2) < r * 0.13
            # Arc mouth: lower half of smaller circle
            mouth_r = r * 0.48
            in_mouth_outer = math.sqrt((dx)**2 + (dy - r*0.05)**2) < mouth_r
            in_mouth_inner = math.sqrt((dx)**2 + (dy - r*0.05)**2) < mouth_r * 0.78
            in_mouth = in_mouth_outer and not in_mouth_inner and dy > r * 0.15
            if in_face:
                img[y, x] = [0.95, 0.82, 0.12, 1.0]
            if in_eye_l or in_eye_r:
                img[y, x] = [0.10, 0.10, 0.10, 1.0]
            if in_mouth:
                img[y, x] = [0.15, 0.10, 0.10, 1.0]
    return img


def draw_lizard(size=TARGET_SIZE):
    img = _blank(size)
    cx, cy = size // 2, size // 2
    body_rx, body_ry = size * 0.22, size * 0.14
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            # Body ellipse
            in_body = (dx / body_rx) ** 2 + (dy / body_ry) ** 2 < 1.0
            # Head
            in_head = math.sqrt((dx - size*0.25)**2 + (dy)**2) < size * 0.12
            # Tail (taper)
            in_tail = (dx < -size*0.20 and abs(dy) < size * 0.08 * (1 + (dx + size*0.20) / (size*0.25)))
            # Legs (4 small rectangles)
            legs = (
                (abs(dx + size*0.10) < size*0.035 and abs(dy - size*0.17) < size*0.09) or
                (abs(dx + size*0.10) < size*0.035 and abs(dy + size*0.17) < size*0.09) or
                (abs(dx - size*0.10) < size*0.035 and abs(dy - size*0.17) < size*0.09) or
                (abs(dx - size*0.10) < size*0.035 and abs(dy + size*0.17) < size*0.09)
            )
            if in_body or in_head or in_tail or legs:
                img[y, x] = [0.18, 0.68, 0.25, 1.0]
    return img


TARGETS = {
    "heart":  draw_heart,
    "smiley": draw_smiley,
    "lizard": draw_lizard,
}


def make_target_tensor(name: str, device, grid: int = GRID):
    """Returns a (4, grid, grid) float32 tensor with the target RGBA centred."""
    small = TARGETS[name]()  # (TARGET_SIZE, TARGET_SIZE, 4)
    canvas = np.zeros((grid, grid, 4), dtype=np.float32)
    oy = (grid - TARGET_SIZE) // 2
    ox = (grid - TARGET_SIZE) // 2
    canvas[oy:oy+TARGET_SIZE, ox:ox+TARGET_SIZE] = small
    t = torch.from_numpy(canvas).permute(2, 0, 1).to(device)
    return t


# ── CA forward pass ────────────────────────────────────────────────────────────

def build_sobel_kernels(device):
    # Depthwise: shape (N, 1, 3, 3) for F.conv2d(groups=N)
    kx = KX.to(device).view(1, 1, 3, 3).expand(N, 1, 3, 3).contiguous()
    ky = KY.to(device).view(1, 1, 3, 3).expand(N, 1, 3, 3).contiguous()
    return kx, ky


def ca_step(state, fc1, fc2, b1, b2, sobel_x, sobel_y):
    """
    One CA step.
    state: (B, N, GRID, GRID)
    Returns: new state, same shape.
    """
    # 1. Perception
    identity = state
    grad_x   = F.conv2d(state, sobel_x, padding=1, groups=N)
    grad_y   = F.conv2d(state, sobel_y, padding=1, groups=N)
    perc     = torch.cat([identity, grad_x, grad_y], dim=1)  # (B, 3N, G, G)

    # 2. MLP (operates cell-wise)
    B, C, G, _ = perc.shape
    flat     = perc.permute(0, 2, 3, 1).reshape(-1, C)  # (B*G*G, 48)

    fc1_q    = fake_quantize(fc1)  # (H, 48)
    fc2_q    = fake_quantize(fc2)  # (N, H)
    hidden   = F.relu(flat @ fc1_q.T + b1)  # (B*G*G, H)
    delta    = (hidden @ fc2_q.T + b2).reshape(B, G, G, N).permute(0, 3, 1, 2)  # (B,N,G,G)

    # 3. Alive mask: max alpha in 3×3 neighbourhood > threshold
    alpha      = state[:, 3:4]
    alive_pre  = F.max_pool2d(alpha, 3, stride=1, padding=1) > ALIVE_THRESH

    # 4. Stochastic update (50%)
    rand_mask = torch.rand_like(alpha) > 0.5

    new_state = state + delta * alive_pre.float() * rand_mask.float()

    # 5. Post-alive mask: dead cells can't spontaneously come alive
    alive_post = F.max_pool2d(new_state[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESH
    new_state  = new_state * alive_post.float()

    return new_state.clamp(-1.0, 1.0)


# ── Loss function ──────────────────────────────────────────────────────────────

def loss_fn(state, target_rgba):
    """
    MSE on all 4 RGBA channels against the target.
    target_rgba has alpha=1 inside the shape and 0 outside, so this single term
    teaches color accuracy, growth inside the boundary, AND death outside it.
    """
    return ((state[:, :4] - target_rgba) ** 2).mean()


# ── Main training loop ─────────────────────────────────────────────────────────

def train(target_name: str, n_steps: int, device):
    print(f"Training target='{target_name}', steps={n_steps}, device={device}")

    kx, ky = build_sobel_kernels(device)

    fc1 = torch.nn.Parameter(torch.randn(H_DIM, N * 3, device=device) * 0.1)
    fc2 = torch.nn.Parameter(torch.zeros(N, H_DIM, device=device))
    b1  = torch.nn.Parameter(torch.zeros(H_DIM, device=device))
    b2  = torch.nn.Parameter(torch.zeros(N, device=device))

    optimizer = torch.optim.Adam([fc1, fc2, b1, b2], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=LR/10)

    G = GRID_TRAIN
    target_t    = make_target_tensor(target_name, device, grid=G)  # (4, G, G)
    target_rgba = target_t[:4].unsqueeze(0)  # (1, 4, G, G) — RGBA, alpha=1 inside shape
    cx = cy = G // 2
    seed_rgba   = target_t[:4, cy, cx]  # 4-vector: RGBA of the seed pixel

    # Pool of POOL_SIZE grids at various growth stages (stored detached on GPU).
    # Sampling from the pool means the model sees early, mid, and late growth
    # every step — no need for long rollouts to train recovery behaviour.
    pool = torch.zeros(POOL_SIZE, N, G, G, device=device)
    pool[:, :4, cy, cx] = seed_rgba  # initialise every pool entry as a fresh seed

    running_loss = 0.0
    import time
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Sample BATCH grids from pool; always include one fresh seed so the
        # model never forgets how to grow from scratch.
        idx = torch.randperm(POOL_SIZE, device=device)[:BATCH]
        state = pool[idx].clone()
        state[0].zero_()
        state[0, :4, cy, cx] = seed_rgba  # fresh seed in slot 0

        # Short rollout — 8-16 steps fits in VRAM without checkpointing.
        n_ca = int(torch.randint(CA_STEPS_MIN, CA_STEPS_MAX + 1, (1,)).item())
        for _ in range(n_ca):
            state = ca_step(state, fc1, fc2, b1, b2, kx, ky)

        # Damage augmentation: zero a random patch in half the batch.
        if step % DAMAGE_EVERY < (DAMAGE_EVERY // 2):
            with torch.no_grad():
                for b in range(DAMAGE_HALF):
                    px = int(torch.randint(DAMAGE_R, G - DAMAGE_R, (1,)).item())
                    py = int(torch.randint(DAMAGE_R, G - DAMAGE_R, (1,)).item())
                    state[b, :, py-DAMAGE_R:py+DAMAGE_R, px-DAMAGE_R:px+DAMAGE_R] = 0.0
            for _ in range(n_ca // 2):
                state = ca_step(state, fc1, fc2, b1, b2, kx, ky)

        # Per-sample loss so we can identify the worst pool entry below.
        per_sample = ((state[:, :4] - target_rgba) ** 2).mean(dim=[1, 2, 3])
        loss = per_sample.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([fc1, fc2, b1, b2], 1.0)
        optimizer.step()
        scheduler.step()

        # Write updated grids back to pool, then replace the worst-performing
        # slot with a fresh seed (Mordvintsev trick). Without this, decayed
        # pool entries accumulate and the model trains on dead grids — which
        # is exactly why low-loss models can still decay during long browser
        # rollouts.
        with torch.no_grad():
            pool[idx] = state.detach()
            worst = per_sample.argmax().item()
            pool[idx[worst]].zero_()
            pool[idx[worst], :4, cy, cx] = seed_rgba

        running_loss = running_loss * 0.99 + loss.item() * 0.01
        if step % 500 == 0 or step == 1:
            elapsed = time.time() - t0
            sps = step / elapsed
            eta  = (n_steps - step) / sps
            print(f"  step {step:6d}/{n_steps}  loss={running_loss:.5f}"
                  f"  {sps:.1f} steps/s  ETA {eta/60:.1f} min")

    print("Training complete.")
    return fc1, fc2, b1, b2


# ── Weight export ──────────────────────────────────────────────────────────────

def export_weights(fc1, fc2, b1, b2, target_name: str, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # fc1: shape (H=96, P=48) → PNG 48 cols × 96 rows
    fc1_np, fc1_min, fc1_scale = quantize_to_uint8(fc1)   # (96, 48)
    img_fc1 = Image.fromarray(fc1_np, mode="L")
    img_fc1.save(out_dir / f"ca_fc1_{target_name}.png")
    print(f"Saved ca_fc1_{target_name}.png  shape={fc1_np.shape}  range=[{fc1_min:.4f}, {fc1_min+fc1_scale:.4f}]")

    # fc2: shape (N=16, H=96) → PNG 96 cols × 16 rows
    fc2_np, fc2_min, fc2_scale = quantize_to_uint8(fc2)   # (16, 96)
    img_fc2 = Image.fromarray(fc2_np, mode="L")
    img_fc2.save(out_dir / f"ca_fc2_{target_name}.png")
    print(f"Saved ca_fc2_{target_name}.png  shape={fc2_np.shape}  range=[{fc2_min:.4f}, {fc2_min+fc2_scale:.4f}]")

    meta = {
        "target":    target_name,
        "fc1_min":   fc1_min,
        "fc1_scale": fc1_scale,
        "fc1_bias":  b1.detach().cpu().numpy().tolist(),
        "fc2_min":   fc2_min,
        "fc2_scale": fc2_scale,
        "fc2_bias":  b2.detach().cpu().numpy().tolist(),
    }
    meta_path = out_dir / f"ca_weights_{target_name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")

    # Also save a full-resolution preview of what the target looks like
    target_img = TARGETS[target_name](TARGET_SIZE)  # (40,40,4)
    rgba_uint8 = (target_img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgba_uint8, mode="RGBA").save(out_dir / f"target_{target_name}.png")
    print(f"Saved target_{target_name}.png")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGETS), default="heart")
    parser.add_argument("--steps",  type=int, default=TRAIN_STEPS)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    if not args.no_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU (pass --no-cuda to suppress this message if intentional)")

    fc1, fc2, b1, b2 = train(args.target, args.steps, device)
    export_weights(fc1, fc2, b1, b2, args.target, pathlib.Path("data"))


if __name__ == "__main__":
    main()
