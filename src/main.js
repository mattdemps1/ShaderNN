// Startup banner — if you don't see this in DevTools console, your browser
// is serving cached JS. Hard-refresh (Ctrl+Shift+R / Cmd+Shift+R).
console.log("%cShaderNN main.js v6 loaded", "color:#9bd3ff;font-weight:bold");

import { CFG } from "./config.js";
import { buildSim } from "./sim.js";
import { buildProjection } from "./projection.js";
import { buildReadout } from "./readout.js";
import { loadMNIST } from "./mnist.js";
import { buildDraw } from "./draw.js";
import { preprocess } from "./preprocess.js";
import {
  saveBin, loadBin, loadDefaultBin,
  saveLocal, loadLocal, clearLocal,
} from "./persist.js";

const params = new URLSearchParams(location.search);
const trainMode = params.has("train");
if (trainMode) CFG.trainBatchPerFrame = 6;

const brainCanvas = document.getElementById("brain");
brainCanvas.width = 512;
brainCanvas.height = 512;

const sim = await buildSim(brainCanvas);
const featureDim = sim.featureDim;
const combinedDim = featureDim + 784;

const projection = buildProjection(sim.W * sim.H, 784, CFG.inputFanIn);
const readout = buildReadout(combinedDim, CFG.numClasses);
const mnist = await loadMNIST();

const drawCanvas = document.getElementById("draw");
const drawer = buildDraw(drawCanvas);
const barsCtx     = document.getElementById("bars").getContext("2d");
const previewCtx  = document.getElementById("preview").getContext("2d");
const previewLbl  = document.getElementById("preview-label");
const curveCtx    = document.getElementById("curve").getContext("2d");
const lossCtx     = document.getElementById("losscurve").getContext("2d");
const popCtx      = document.getElementById("poprate").getContext("2d");
previewCtx.imageSmoothingEnabled = false;
const previewImageData = previewCtx.createImageData(28, 28);

// HUD elements
const $ = id => document.getElementById(id);
const hudStep   = $("hud-step");
const hudTrain  = $("hud-train");
const hudVal    = $("hud-val");
const hudBest   = $("hud-best");
const hudLoss   = $("hud-loss");
const hudTput   = $("hud-tput");
const hudPop    = $("hud-pop");
const hudActive = $("hud-active");
const hudTmean  = $("hud-tmean");
const hudTstd   = $("hud-tstd");
const hudPmean  = $("hud-pmean");
const hudConf   = $("hud-conf");
const hudHact   = $("hud-hact");
const hudDim    = $("hud-dim");

const btnSave   = $("btn-save");
const btnLoad   = $("btn-load");
const btnReset  = $("btn-reset");
const btnRecal  = $("btn-recal");
const fileLoad  = $("file-load");

let exampleCount = 0;
let paused   = false;
let learning = true;
let viewMode = 0;
let trainAccEMA = 0.1;
let valAcc = null;
let bestVal = 0;
let popRate = 0;
let activeNeurons = 0;
let traceMean = 0, traceStd = 0;
let pixelMean = 0;
let livePred = null;
let lastLiveImage = null;

const valHistory  = [];
const lossHistory = [];
const popHistory  = [];
const HIST_MAX = 200;

// Throughput (EMA of examples per second)
let throughputEMA = 0;
let lastThroughputAt = performance.now();
let exSinceThroughput = 0;

let pixelBlockScale = 0.5;
const combinedFeatures = new Float32Array(combinedDim);

function featurize(image28, time) {
  const drive = projection.project(image28);
  const f = sim.present(drive, time);

  // Reservoir block: L2-normalize in place into combinedFeatures[0..D)
  let sqR = 0, sumR = 0;
  for (let i = 0; i < featureDim; i++) { const v = f[i]; sqR += v * v; sumR += v; }
  const invR = sqR > 1e-12 ? 1 / Math.sqrt(sqR) : 0;
  for (let i = 0; i < featureDim; i++) combinedFeatures[i] = f[i] * invR;

  // Pixel block: L2-normalize, scale, into [D..D+784)
  let sqP = 0, sumP = 0;
  for (let i = 0; i < 784; i++) { const v = image28[i]; sqP += v * v; sumP += v; }
  const invP = sqP > 1e-12 ? pixelBlockScale / Math.sqrt(sqP) : 0;
  for (let i = 0; i < 784; i++) combinedFeatures[featureDim + i] = image28[i] * invP;

  // Diagnostics: rolling stats on the raw (pre-normalization) blocks
  const meanR = sumR / featureDim;
  let varR = 0;
  for (let i = 0; i < featureDim; i++) { const d = f[i] - meanR; varR += d * d; }
  const stdR = Math.sqrt(varR / featureDim);
  traceMean = traceMean * 0.95 + meanR * 0.05;
  traceStd  = traceStd  * 0.95 + stdR  * 0.05;
  pixelMean = pixelMean * 0.95 + (sumP / 784) * 0.05;

  return combinedFeatures;
}

// Initial weight load: localStorage (IndexedDB) → bundled .bin → freshly-init MLP
if (await loadLocal(readout)) {
  console.log("restored readout weights from IndexedDB");
  trainAccEMA = 0.5;
} else if (await loadDefaultBin(readout)) {
  console.log("loaded bundled default weights (.bin)");
  trainAccEMA = 0.5;
}

// One-shot vThr calibration
{
  const sample = mnist.getTest(0).image;
  for (let attempt = 0; attempt < 6; attempt++) {
    featurize(sample, performance.now() * 0.001);
    const r = sim.populationRate();
    if (r >= 0.05 && r <= 0.25) {
      console.log(`reservoir populationRate=${(r*100).toFixed(1)}%, vThr=${CFG.vThr.toFixed(3)} (calibrated)`);
      popRate = r;
      activeNeurons = Math.round(r * sim.W * sim.H);
      break;
    }
    const next = r > 0.25 ? CFG.vThr + 0.05 : CFG.vThr - 0.05;
    console.log(`reservoir popRate=${(r*100).toFixed(1)}% out of [5%,25%] — adjusting vThr ${CFG.vThr.toFixed(3)} → ${next.toFixed(3)}`);
    CFG.vThr = next;
  }
}

// One-time diagnostic dump after warmup
let dumpedDiag = false;
function maybeDumpDiagnostics() {
  if (dumpedDiag || exampleCount < 8) return;
  dumpedDiag = true;
  console.group("ShaderNN diagnostics (after warmup)");
  console.log(`feature dim: ${combinedDim} (reservoir ${featureDim} + pixels 784)`);
  console.log(`readout: ${readout.D}→${readout.H}→${readout.C} (MLP, Adam lr=${CFG.readoutLR})`);
  console.log(`trace mean/std: ${traceMean.toFixed(4)} / ${traceStd.toFixed(4)}`);
  console.log(`pixel mean: ${pixelMean.toFixed(4)}`);
  console.log(`population rate: ${(popRate*100).toFixed(1)}%`);
  console.log(`hidden activation rate: ${(readout.stats.hiddenActiveRate*100).toFixed(1)}%`);
  console.log(`loss EMA: ${readout.stats.lossEMA.toFixed(3)}`);
  if (traceStd < 1e-4) console.warn("  ⚠ trace std near zero — reservoir may not be responding to input");
  if (readout.stats.hiddenActiveRate < 0.05) console.warn("  ⚠ hidden layer mostly dead — try lowering lr or check feature scaling");
  if (readout.stats.hiddenActiveRate > 0.95) console.warn("  ⚠ hidden layer always firing — try lowering lr or weight init");
  console.groupEnd();
}

// ── Keyboard ─────────────────────────────────────────────────────────
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { paused = !paused; setToggle("c-pause", paused); e.preventDefault(); }
  if (e.code === "KeyC")  { drawer.clear(); livePred = null; lastLiveImage = null; }
  if (e.code === "KeyR")  { fullReset(); }
  if (e.code === "KeyL")  { learning = !learning; setToggle("c-freeze", !learning); }
  if (e.code === "KeyW")  { viewMode = (viewMode + 1) % 3; }
});

function setToggle(id, on) {
  const el = $(id);
  if (el) el.checked = on;
}

function fullReset() {
  sim.reset();
  readout.reset();
  clearLocal();
  exampleCount = 0;
  trainAccEMA = 0.1;
  valAcc = null; bestVal = 0;
  valHistory.length = 0;
  lossHistory.length = 0;
  popHistory.length = 0;
  dumpedDiag = false;
}

// ── Buttons ──────────────────────────────────────────────────────────
btnSave.addEventListener("click", () => saveBin(readout));
btnLoad.addEventListener("click", () => fileLoad.click());
fileLoad.addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    await loadBin(file, readout);
    await saveLocal(readout);
    console.log("loaded weights from", file.name);
  } catch (err) {
    alert("failed to load weights: " + err.message);
  }
  fileLoad.value = "";
});
btnReset.addEventListener("click", () => {
  if (confirm("reset readout weights and clear training history?")) fullReset();
});
btnRecal.addEventListener("click", () => {
  sim.reset();
  console.log("reservoir re-rolled (random weights regenerated, readout untouched)");
});

// ── Sliders ──────────────────────────────────────────────────────────
function bindSlider(id, valId, getter, setter, fmt = v => v.toFixed(3)) {
  const el = $(id), v = $(valId);
  if (!el) return;
  const sync = () => { v.textContent = fmt(getter()); };
  el.value = String(getter());
  sync();
  el.addEventListener("input", () => {
    setter(parseFloat(el.value));
    sync();
  });
}
bindSlider("c-vthr",   "v-vthr",   () => CFG.vThr,            v => CFG.vThr = v);
bindSlider("c-drive",  "v-drive",  () => CFG.inputDriveScale, v => CFG.inputDriveScale = v, v => v.toFixed(2));
bindSlider("c-tdecay", "v-tdecay", () => CFG.traceDecay,      v => CFG.traceDecay = v);
bindSlider("c-noise",  "v-noise",  () => CFG.noiseRate,       v => CFG.noiseRate = v, v => v.toFixed(4));
bindSlider("c-vnoise", "v-vnoise", () => CFG.vNoise,          v => CFG.vNoise = v, v => v.toFixed(3));
bindSlider("c-lr",     "v-lr",     () => CFG.readoutLR,       v => CFG.readoutLR = v, v => v.toExponential(1));
bindSlider("c-pxs",    "v-pxs",    () => pixelBlockScale,     v => pixelBlockScale = v, v => v.toFixed(2));
bindSlider("c-bpf",    "v-bpf",    () => CFG.trainBatchPerFrame, v => CFG.trainBatchPerFrame = (v|0), v => String(v|0));

$("c-aug").addEventListener("change", e => mnist.setAugment(e.target.checked));
$("c-pause").addEventListener("change", e => paused = e.target.checked);
$("c-freeze").addEventListener("change", e => learning = !e.target.checked);

// Silent mode: zero spontaneous-activity sources so the reservoir is fully
// quiescent unless driven by input. We snapshot the user's current values
// when the box is checked so unchecking restores them — including any
// changes they made via the sliders.
let savedNoise = { noiseRate: CFG.noiseRate, vNoise: CFG.vNoise };
$("c-silent").addEventListener("change", e => {
  if (e.target.checked) {
    savedNoise = { noiseRate: CFG.noiseRate, vNoise: CFG.vNoise };
    CFG.noiseRate = 0; CFG.vNoise = 0;
  } else {
    CFG.noiseRate = savedNoise.noiseRate;
    CFG.vNoise    = savedNoise.vNoise;
  }
  // Refresh the slider readouts so they reflect the swap
  $("v-noise").textContent  = CFG.noiseRate.toFixed(4);
  $("v-vnoise").textContent = CFG.vNoise.toFixed(3);
  $("c-noise").value  = String(CFG.noiseRate);
  $("c-vnoise").value = String(CFG.vNoise);
});

// ── Training & inference ─────────────────────────────────────────────
function evalVal() {
  let correct = 0;
  for (let i = 0; i < CFG.valBatch; i++) {
    const idx = (Math.random() * mnist.numTest) | 0;
    const ex = mnist.getTest(idx);
    const feat = featurize(ex.image, performance.now() * 0.001 + i);
    const { label } = readout.predict(feat);
    if (label === ex.label) correct++;
  }
  return correct / CFG.valBatch;
}

let lastPopReadAt = 0;
function trainStep(t) {
  const ex = mnist.nextTrain();
  const feat = featurize(ex.image, t * 0.001);
  const result = readout.train(feat, ex.label);
  trainAccEMA = trainAccEMA * (1 - CFG.accEMA) + (result.correct ? 1 : 0) * CFG.accEMA;
  exampleCount++;
  exSinceThroughput++;

  if (exampleCount % CFG.valEvery === 0) {
    valAcc = evalVal();
    valHistory.push(valAcc);
    while (valHistory.length > HIST_MAX) valHistory.shift();
    lossHistory.push(readout.stats.lossEMA);
    while (lossHistory.length > HIST_MAX) lossHistory.shift();
    if (valAcc > bestVal) {
      bestVal = valAcc;
      saveLocal(readout);
    }
  }

  if (t - lastPopReadAt > 800) {
    popRate = sim.populationRate();
    activeNeurons = Math.round(popRate * sim.W * sim.H);
    popHistory.push(popRate);
    while (popHistory.length > HIST_MAX) popHistory.shift();
    lastPopReadAt = t;
  }
}

let lastPreviewImage = null;
function liveInfer(t) {
  if (drawer.isEmpty()) {
    livePred = null;
    lastLiveImage = null;
    if (lastPreviewImage !== null) {
      // Clear the preview when the canvas is empty
      previewCtx.fillStyle = "#000";
      previewCtx.fillRect(0, 0, 72, 72);
      previewLbl.textContent = "—";
      lastPreviewImage = null;
    }
    return;
  }
  let changed = false;
  if (!lastLiveImage || lastLiveImage.length !== drawer.image.length) {
    lastLiveImage = new Float32Array(drawer.image.length);
    changed = true;
  } else {
    for (let i = 0; i < drawer.image.length; i++) {
      if (Math.abs(drawer.image[i] - lastLiveImage[i]) > 1e-3) { changed = true; break; }
    }
  }
  if (!changed) return;
  lastLiveImage.set(drawer.image);

  const cleaned = preprocess(drawer.image);
  const feat = featurize(cleaned, t * 0.001 + 9999);
  const pred = readout.predict(feat);
  livePred = { label: pred.label, probs: Float32Array.from(pred.probs) };
  drawPreview(cleaned, pred.label);
  lastPreviewImage = cleaned;
}

function drawPreview(image28, label) {
  // image28 is the 28×28 the network actually classified (post-preprocess).
  // Render it large and pixelated so you can see exactly what the model saw.
  const data = previewImageData.data;
  for (let i = 0; i < 28 * 28; i++) {
    const v = Math.min(255, Math.max(0, Math.round(image28[i] * 255)));
    data[i * 4]     = v;
    data[i * 4 + 1] = v;
    data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  // Stamp into a tmp canvas at native resolution, then scale up via drawImage.
  // (createImageData → putImageData is 1:1 size, so we need an intermediate.)
  const tmpCv = drawPreview._tmp || (drawPreview._tmp = document.createElement("canvas"));
  tmpCv.width = 28; tmpCv.height = 28;
  tmpCv.getContext("2d").putImageData(previewImageData, 0, 0);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, 72, 72);
  previewCtx.drawImage(tmpCv, 0, 0, 72, 72);
  previewLbl.textContent = `pred: ${label}`;
}

// ── Drawing functions ────────────────────────────────────────────────
function drawBars() {
  const w = barsCtx.canvas.width, h = barsCtx.canvas.height;
  barsCtx.fillStyle = "#0e0e14"; barsCtx.fillRect(0, 0, w, h);
  const probs = livePred ? livePred.probs : new Float32Array(10);
  const top   = livePred ? livePred.label : -1;
  const padL = 28, padR = 8, padT = 8, padB = 12;
  const rowH = (h - padT - padB) / 10;
  const maxW = w - padL - padR;
  barsCtx.font = "11px ui-monospace, monospace";
  barsCtx.textBaseline = "middle";
  for (let c = 0; c < 10; c++) {
    const y = padT + c * rowH;
    const p = probs[c] || 0;
    const bw = Math.max(1, p * maxW);
    barsCtx.fillStyle = (c === top) ? "#9bd3ff" : "#3a3a4a";
    barsCtx.fillRect(padL, y + 2, bw, rowH - 4);
    barsCtx.fillStyle = (c === top) ? "#fff" : "#7a7a88";
    barsCtx.fillText(String(c), 8, y + rowH / 2);
    barsCtx.fillStyle = "#5a5a68";
    barsCtx.fillText((p * 100).toFixed(0) + "%", padL + bw + 4, y + rowH / 2);
  }
  if (!livePred) {
    barsCtx.fillStyle = "#3a3a48";
    barsCtx.fillText(trainMode ? "training mode (no inference)" : "draw a digit ↑", padL, h - 4);
  }
}

function drawCurve(ctx, hist, color, label, fmtVal, yMax = 1.0, latestVal = null) {
  const w = ctx.canvas.width, h = ctx.canvas.height;
  ctx.fillStyle = "#0e0e14"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#1a1a25"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (const r of [0.25, 0.5, 0.75, 1.0]) {
    const y = h - r * (h - 18) - 2;
    ctx.moveTo(0, y); ctx.lineTo(w, y);
  }
  ctx.stroke();
  if (hist.length > 1) {
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    hist.forEach((v, i) => {
      const x = (i / (HIST_MAX - 1)) * w;
      const y = h - (v / yMax) * (h - 18) - 2;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#fff"; ctx.font = "bold 16px ui-monospace, monospace";
  ctx.fillText(latestVal !== null ? fmtVal(latestVal) : "—", 6, 20);
  ctx.font = "9px ui-monospace, monospace"; ctx.fillStyle = "#5a5a68";
  ctx.fillText(label, 6, h - 4);
}

function drawValCurve() {
  drawCurve(curveCtx, valHistory, "#9bd3ff",
    `val acc (best ${(bestVal*100).toFixed(0)}%)`,
    v => `${(v*100).toFixed(0)}%`, 1.0, valAcc);
}

function drawLossCurve() {
  // Auto-scale loss curve: cap to max(2.5, max-in-history)
  const yMax = Math.max(2.5, ...(lossHistory.length ? lossHistory : [2.5]));
  drawCurve(lossCtx, lossHistory, "#ffb39b", "cross-entropy",
    v => v.toFixed(3), yMax, readout.stats.lossEMA);
}

function drawPopCurve() {
  drawCurve(popCtx, popHistory, "#9bffb1", "pop. rate",
    v => `${(v*100).toFixed(1)}%`, 0.5, popRate);
}

// ── Frame loop ───────────────────────────────────────────────────────
function frame(t) {
  if (!paused && learning) {
    for (let i = 0; i < CFG.trainBatchPerFrame; i++) trainStep(t);
  }
  // Live inference always runs (independent of pause / freeze) so the
  // prediction reflects whatever you've drawn right now.
  if (!trainMode) liveInfer(t);

  sim.visualize(viewMode);
  drawBars();
  drawValCurve();
  drawLossCurve();
  drawPopCurve();

  // Throughput EMA every ~500 ms
  if (t - lastThroughputAt > 500) {
    const dt = (t - lastThroughputAt) / 1000;
    const inst = exSinceThroughput / dt;
    throughputEMA = throughputEMA === 0 ? inst : (throughputEMA * 0.7 + inst * 0.3);
    exSinceThroughput = 0;
    lastThroughputAt = t;
  }

  hudStep.textContent   = exampleCount.toString();
  hudTrain.textContent  = (trainAccEMA * 100).toFixed(1) + "%";
  hudVal.textContent    = valAcc !== null ? (valAcc * 100).toFixed(0) + "%" : "—";
  hudBest.textContent   = bestVal > 0 ? (bestVal * 100).toFixed(0) + "%" : "—";
  hudLoss.textContent   = readout.stats.lossEMA.toFixed(3);
  hudTput.textContent   = throughputEMA.toFixed(0) + " ex/s";
  hudPop.textContent    = (popRate * 100).toFixed(1) + "%";
  hudActive.textContent = activeNeurons.toString();
  hudTmean.textContent  = traceMean.toExponential(2);
  hudTstd.textContent   = traceStd.toExponential(2);
  hudPmean.textContent  = pixelMean.toFixed(3);
  hudConf.textContent   = (readout.stats.conf * 100).toFixed(0) + "%";
  hudHact.textContent   = (readout.stats.hiddenActiveRate * 100).toFixed(0) + "%";
  hudDim.textContent    = `${readout.D}→${readout.H}→${readout.C}`;

  maybeDumpDiagnostics();

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
