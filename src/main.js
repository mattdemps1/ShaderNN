import { CFG } from "./config.js";
import { buildSim } from "./sim.js";
import { buildProjection } from "./projection.js";
import { buildReadout } from "./readout.js";
import { loadMNIST } from "./mnist.js";
import { buildDraw } from "./draw.js";
import { preprocess } from "./preprocess.js";
import { savePNG, loadPNG, loadDefaultPNG, saveLocal, loadLocal, clearLocal } from "./persist.js";

const params = new URLSearchParams(location.search);
const trainMode = params.has("train");
if (trainMode) CFG.trainBatchPerFrame = 6;

const brainCanvas = document.getElementById("brain");
brainCanvas.width = 512;
brainCanvas.height = 512;

const sim = await buildSim(brainCanvas);
const featureDim = sim.featureDim;     // 2 × W × H
const combinedDim = featureDim + 784;  // reservoir features + raw preprocessed pixels

const projection = buildProjection(sim.W * sim.H, 784, CFG.inputFanIn);
const readout = buildReadout(combinedDim, CFG.numClasses);
const mnist = await loadMNIST();

const drawCanvas = document.getElementById("draw");
const drawer = buildDraw(drawCanvas);
const barsCtx = document.getElementById("bars").getContext("2d");
const curveCtx = document.getElementById("curve").getContext("2d");

const hudStep  = document.getElementById("hud-step");
const hudTrain = document.getElementById("hud-train");
const hudVal   = document.getElementById("hud-val");
const hudPop   = document.getElementById("hud-pop");
const btnSave  = document.getElementById("btn-save");
const btnLoad  = document.getElementById("btn-load");
const fileLoad = document.getElementById("file-load");

let exampleCount = 0;
let paused   = false;
let learning = true;
let viewMode = 0;
let trainAccEMA = 0.1;
let valAcc = null;
let bestVal = 0;
let popRate = 0;
let livePred = null;
let lastLiveImage = null;
const valHistory = [];
const VAL_HIST_MAX = 200;

const combinedFeatures = new Float32Array(combinedDim);
function featurize(image28, time) {
  const drive = projection.project(image28);
  const f = sim.present(drive, time);
  combinedFeatures.set(f, 0);
  combinedFeatures.set(image28, featureDim);
  return combinedFeatures;
}

if (loadLocal(readout)) {
  console.log("restored readout weights from localStorage");
  trainAccEMA = 0.5;
} else if (await loadDefaultPNG(readout)) {
  console.log("loaded bundled default weights");
  trainAccEMA = 0.5;
}

window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { paused = !paused; e.preventDefault(); }
  if (e.code === "KeyC") { drawer.clear(); livePred = null; lastLiveImage = null; }
  if (e.code === "KeyR") {
    sim.reset();
    readout.reset();
    clearLocal();
    exampleCount = 0;
    trainAccEMA = 0.1;
    valAcc = null; bestVal = 0;
    valHistory.length = 0;
  }
  if (e.code === "KeyL") learning = !learning;
  if (e.code === "KeyW") viewMode = (viewMode + 1) % 3;
});

btnSave.addEventListener("click", () => savePNG(readout));
btnLoad.addEventListener("click", () => fileLoad.click());
fileLoad.addEventListener("change", async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    await loadPNG(file, readout);
    saveLocal(readout);
    console.log("loaded weights from", file.name);
  } catch (err) {
    alert("failed to load weights: " + err.message);
  }
  fileLoad.value = "";
});

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

function trainStep(t) {
  const ex = mnist.nextTrain();
  const feat = featurize(ex.image, t * 0.001);
  const result = readout.train(feat, ex.label);
  trainAccEMA = trainAccEMA * (1 - CFG.accEMA) + (result.correct ? 1 : 0) * CFG.accEMA;
  exampleCount++;
  if (exampleCount % CFG.valEvery === 0) {
    valAcc = evalVal();
    valHistory.push(valAcc);
    while (valHistory.length > VAL_HIST_MAX) valHistory.shift();
    if (valAcc > bestVal) {
      bestVal = valAcc;
      saveLocal(readout);
    }
  }
  popRate = sim.populationRate();
}

function liveInfer(t) {
  if (drawer.isEmpty()) {
    livePred = null;
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
}

function drawBars() {
  const w = barsCtx.canvas.width, h = barsCtx.canvas.height;
  barsCtx.fillStyle = "#0e0e14";
  barsCtx.fillRect(0, 0, w, h);

  const probs = livePred ? livePred.probs : new Float32Array(10);
  const top = livePred ? livePred.label : -1;

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

function drawCurve() {
  const w = curveCtx.canvas.width, h = curveCtx.canvas.height;
  curveCtx.fillStyle = "#0e0e14";
  curveCtx.fillRect(0, 0, w, h);
  curveCtx.strokeStyle = "#1a1a25";
  curveCtx.lineWidth = 1;
  curveCtx.beginPath();
  for (const r of [0.25, 0.5, 0.75, 1.0]) {
    const y = h - r * (h - 18) - 2;
    curveCtx.moveTo(0, y); curveCtx.lineTo(w, y);
  }
  curveCtx.stroke();

  if (valHistory.length > 1) {
    curveCtx.strokeStyle = "#9bd3ff";
    curveCtx.lineWidth = 2;
    curveCtx.beginPath();
    valHistory.forEach((v, i) => {
      const x = (i / (VAL_HIST_MAX - 1)) * w;
      const y = h - v * (h - 18) - 2;
      if (i === 0) curveCtx.moveTo(x, y); else curveCtx.lineTo(x, y);
    });
    curveCtx.stroke();
  }
  curveCtx.fillStyle = "#fff";
  curveCtx.font = "bold 18px ui-monospace, monospace";
  curveCtx.fillText(valAcc !== null ? `${(valAcc * 100).toFixed(0)}%` : "--", 6, 22);
  curveCtx.font = "9px ui-monospace, monospace";
  curveCtx.fillStyle = "#5a5a68";
  const hint = bestVal > 0.85 ? `  ← click 'save weights' to bundle as default` : "";
  curveCtx.fillText(`val acc (every ${CFG.valEvery} examples)  best: ${(bestVal*100).toFixed(0)}%${hint}`, 6, h - 4);
}

function frame(t) {
  if (!paused) {
    if (learning) {
      for (let i = 0; i < CFG.trainBatchPerFrame; i++) trainStep(t);
    }
    if (!trainMode) liveInfer(t);
  }

  sim.visualize(viewMode);
  drawBars();
  drawCurve();
  hudStep.textContent  = exampleCount.toString();
  hudTrain.textContent = (trainAccEMA * 100).toFixed(1) + "%";
  hudVal.textContent   = valAcc !== null ? (valAcc * 100).toFixed(0) + "%" : "—";
  hudPop.textContent   = (popRate * 100).toFixed(1) + "%";

  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
