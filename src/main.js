import { CFG } from "./config.js";
import { buildCA } from "./ca.js";
import { loadWeights } from "./weights.js";

// Seed colours per target (RGBA float, normalised 0-1).
// Must match draw_* functions in build_ca.py — these are the centre pixel of
// the target image, used both as the persistent seed and as the colour the
// model expects to see at training time.
const SEED_COLORS = {
  heart:  [0.87, 0.18, 0.18, 1.0],
  smiley: [0.95, 0.82, 0.12, 1.0],
  lizard: [0.18, 0.68, 0.25, 1.0],
};

// After pointerup, re-stamp the seed for this many frames so the pattern
// can bootstrap even if the first stamp gets a negative-alpha delta.
const RESEED_FRAMES = 30;

async function main() {
  const canvas  = document.getElementById("ca-canvas");
  canvas.width  = CFG.gridW;
  canvas.height = CFG.gridH;

  const statusEl = document.getElementById("status");
  statusEl.textContent = "Loading weights…";

  let ca;
  try {
    ca = buildCA(canvas);
  } catch (e) {
    statusEl.textContent = `WebGL error: ${e.message}`;
    throw e;
  }

  let currentTarget = CFG.targets[0];
  ca.reset(SEED_COLORS[currentTarget] ?? [1, 1, 1, 1]);

  function updateWeightImages(target) {
    const dir = CFG.dataDir;
    document.getElementById("img-fc1").src = `${dir}/ca_fc1_${target}.png`;
    document.getElementById("img-fc2").src = `${dir}/ca_fc2_${target}.png`;
  }

  async function reloadWeights(target) {
    statusEl.textContent = "Loading weights…";
    try {
      const { ubo } = await loadWeights(ca.gl, target);
      ca.bindWeights(ubo);
      updateWeightImages(target);
      statusEl.textContent = "Ready";
      return true;
    } catch (e) {
      statusEl.textContent = `Weights not found — run: python build_ca.py --target ${target}`;
      console.warn("Weight load failed:", e);
      return false;
    }
  }

  let weightsLoaded = await reloadWeights(currentTarget);

  let paused      = false;
  let frameIdx    = 0;
  let stepsFrame  = CFG.stepsPerFrame;
  let lastT       = performance.now();
  let fpsEma      = 60;
  let isDamaging  = false;
  let reseedTimer = 0;

  const hudStep  = document.getElementById("hud-step");
  const hudFps   = document.getElementById("hud-fps");
  const selTarget = document.getElementById("sel-target");
  const spdBtns  = document.querySelectorAll("[data-spf]");

  spdBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      stepsFrame = parseInt(btn.dataset.spf, 10);
      spdBtns.forEach(b => b.classList.toggle("active", b === btn));
    });
  });
  spdBtns.forEach(b => b.classList.toggle("active", parseInt(b.dataset.spf, 10) === stepsFrame));

  selTarget.value = currentTarget;
  selTarget.addEventListener("change", async () => {
    currentTarget = selTarget.value;
    ca.reset(SEED_COLORS[currentTarget] ?? [1, 1, 1, 1]);
    frameIdx = 0;
    weightsLoaded = await reloadWeights(currentTarget);
  });

  window.addEventListener("keydown", e => {
    if (e.code === "Space") { paused = !paused; e.preventDefault(); }
    if (e.code === "KeyR")  { ca.reset(SEED_COLORS[currentTarget] ?? [1,1,1,1]); frameIdx = 0; }
  });

  function canvasUV(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left)  / rect.width,
      1.0 - (e.clientY - rect.top) / rect.height,
    ];
  }

  canvas.addEventListener("pointerdown", e => {
    isDamaging = true;
    canvas.setPointerCapture(e.pointerId);
    ca.setDamage(...canvasUV(e));
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", e => {
    if (!isDamaging) return;
    ca.setDamage(...canvasUV(e));
  });
  canvas.addEventListener("pointerup",  () => { if (isDamaging) reseedTimer = RESEED_FRAMES; isDamaging = false; ca.clearDamage(); });
  canvas.addEventListener("pointerout", () => { if (isDamaging) reseedTimer = RESEED_FRAMES; isDamaging = false; ca.clearDamage(); });

  function frame(now) {
    if (!paused && weightsLoaded) {
      if (reseedTimer > 0) {
        ca.seed(SEED_COLORS[currentTarget] ?? [1,1,1,1]);
        reseedTimer--;
      }
      for (let i = 0; i < stepsFrame; i++) ca.step();
      frameIdx += stepsFrame;
    }
    ca.display();

    const dt = Math.max(now - lastT, 1);
    lastT = now;
    fpsEma = fpsEma * 0.92 + (1000 / dt) * 0.08;
    hudStep.textContent = frameIdx;
    hudFps.textContent  = fpsEma.toFixed(0);

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch(e => {
  console.error(e);
  const s = document.getElementById("status");
  if (s) s.textContent = `Fatal: ${e.message}`;
});
