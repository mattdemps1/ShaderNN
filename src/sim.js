import { CFG } from "./config.js";
import {
  getGL, program, quad, makeFloatTex, makeFBO, bindTexUnits, fetchText,
} from "./glutil.js";

// Box-Muller standard normal sample
function gauss() {
  const u = Math.max(1e-9, Math.random());
  const v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Build 6 weight textures (3 local + 3 long-range) of fixed signed sparse random
// values. Each pixel packs a 9-element 3×3 kernel into RGBA channels of 3
// textures (last channel of texture #2 is unused).
function makeReservoirWeights(W, H, sigma, sparsity) {
  const tex = [
    new Float32Array(W * H * 4),
    new Float32Array(W * H * 4),
    new Float32Array(W * H * 4),
  ];
  for (let i = 0; i < W * H; i++) {
    for (let slot = 0; slot < 9; slot++) {
      const ti = (slot / 4) | 0;
      const ci = slot - ti * 4;
      const drop = Math.random() < sparsity;
      const v = drop ? 0.0 : gauss() * sigma;
      tex[ti][i * 4 + ci] = v;
    }
    // unused channel
    tex[2][i * 4 + 1] = 0;
    tex[2][i * 4 + 2] = 0;
    tex[2][i * 4 + 3] = 0;
  }
  return tex;
}

export async function buildSim(canvas) {
  const gl = getGL(canvas);
  const W = CFG.grid, H = CFG.grid;

  const [vs, inFS, nrFS, vzFS] = await Promise.all([
    fetchText("./shaders/quad.vert.glsl"),
    fetchText("./shaders/input.frag.glsl"),
    fetchText("./shaders/neuron.frag.glsl"),
    fetchText("./shaders/visualize.frag.glsl"),
  ]);

  const pIn = program(gl, vs, inFS, "input");
  const pNr = program(gl, vs, nrFS, "neuron");
  const pVz = program(gl, vs, vzFS, "visualize");

  const va = quad(gl);

  const initState = () => {
    const arr = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) arr[i * 4] = Math.random() * 0.1;
    return arr;
  };
  const zeros = () => new Float32Array(W * H * 4);

  const stateA = makeFloatTex(gl, W, H, initState());
  const stateB = makeFloatTex(gl, W, H, initState());
  const driveTex = makeFloatTex(gl, W, H, zeros());
  const inputTex = makeFloatTex(gl, W, H, zeros());

  // Fixed reservoir weights (never updated)
  const wLocalData = makeReservoirWeights(W, H, CFG.resSigmaLocal, CFG.resSparsity);
  const wLongData  = makeReservoirWeights(W, H, CFG.resSigmaLong,  CFG.resSparsity);
  const w = [
    makeFloatTex(gl, W, H, wLocalData[0]),
    makeFloatTex(gl, W, H, wLocalData[1]),
    makeFloatTex(gl, W, H, wLocalData[2]),
    makeFloatTex(gl, W, H, wLongData[0]),
    makeFloatTex(gl, W, H, wLongData[1]),
    makeFloatTex(gl, W, H, wLongData[2]),
  ];

  const fboInput  = makeFBO(gl, [inputTex]);
  const fboStateA = makeFBO(gl, [stateA]);
  const fboStateB = makeFBO(gl, [stateB]);

  let stateRead = stateA, stateWrite = stateB;
  let stateReadFBO = fboStateA, stateWriteFBO = fboStateB;

  const draw = () => {
    gl.bindVertexArray(va);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const u = (p, name) => gl.getUniformLocation(p, name);

  // Upload per-neuron drive (Float32Array length W*H, in [0,1]) into the .r channel
  // of driveTex. We pack it as RGBA32F since that's what glutil.makeFloatTex creates.
  const drivePacked = new Float32Array(W * H * 4);
  function setDrive(driveFlat) {
    for (let i = 0; i < W * H; i++) drivePacked[i * 4] = driveFlat[i];
    gl.bindTexture(gl.TEXTURE_2D, driveTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, drivePacked);
  }

  function step(time) {
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    // 1. Input — Poisson-sample from drive texture
    gl.useProgram(pIn);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboInput);
    gl.uniform2f(u(pIn, "u_grid"), W, H);
    gl.uniform1f(u(pIn, "u_driveScale"), CFG.inputDriveScale);
    gl.uniform1f(u(pIn, "u_noiseRate"), CFG.noiseRate);
    gl.uniform1f(u(pIn, "u_time"), time);
    bindTexUnits(gl, pIn, [{ name: "u_drive", tex: driveTex, unit: 0 }]);
    draw();

    // 2. Neuron update — LIF + signed recurrent reservoir
    gl.useProgram(pNr);
    gl.bindFramebuffer(gl.FRAMEBUFFER, stateWriteFBO);
    gl.uniform2f(u(pNr, "u_grid"), W, H);
    gl.uniform1f(u(pNr, "u_leak"), CFG.leak);
    gl.uniform1f(u(pNr, "u_vThr"), CFG.vThr);
    gl.uniform1f(u(pNr, "u_vReset"), CFG.vReset);
    gl.uniform1f(u(pNr, "u_tRef"), CFG.tRef);
    gl.uniform1f(u(pNr, "u_traceDecay"), CFG.traceDecay);
    gl.uniform1f(u(pNr, "u_thrJitter"), CFG.thrJitter);
    gl.uniform1f(u(pNr, "u_vNoise"), CFG.vNoise);
    gl.uniform1f(u(pNr, "u_longStride"), CFG.longStride);
    gl.uniform1f(u(pNr, "u_time"), time);
    bindTexUnits(gl, pNr, [
      { name: "u_state", tex: stateRead, unit: 0 },
      { name: "u_input", tex: inputTex,  unit: 1 },
      { name: "u_w0",    tex: w[0],      unit: 2 },
      { name: "u_w1",    tex: w[1],      unit: 3 },
      { name: "u_w2",    tex: w[2],      unit: 4 },
      { name: "u_w3",    tex: w[3],      unit: 5 },
      { name: "u_w4",    tex: w[4],      unit: 6 },
      { name: "u_w5",    tex: w[5],      unit: 7 },
    ]);
    draw();

    [stateRead, stateWrite] = [stateWrite, stateRead];
    [stateReadFBO, stateWriteFBO] = [stateWriteFBO, stateReadFBO];
  }

  // Run a full presentation: clear state, set drive, run presentSteps. Takes
  // two trace snapshots (mid + end) and concatenates → 2*W*H features. The two
  // snapshots capture different phases of reservoir dynamics for the linear
  // readout to combine.
  const stateBuf  = new Float32Array(W * H * 4);
  const featBuf   = new Float32Array(W * H * 2);
  const zeroState = new Float32Array(W * H * 4);
  function present(driveFlat, time) {
    gl.bindTexture(gl.TEXTURE_2D, stateRead);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, zeroState);
    setDrive(driveFlat);
    for (let s = 0; s < CFG.presentSteps; s++) {
      step(time + s * 0.0137);
      if (s + 1 === CFG.midSnapshotStep) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, stateReadFBO);
        gl.readBuffer(gl.COLOR_ATTACHMENT0);
        gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, stateBuf);
        for (let i = 0; i < W * H; i++) featBuf[i] = stateBuf[i * 4 + 2];
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, stateReadFBO);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, stateBuf);
    for (let i = 0; i < W * H; i++) featBuf[W * H + i] = stateBuf[i * 4 + 2];
    return featBuf;
  }

  function readTraces() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, stateReadFBO);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, stateBuf);
    const out = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) out[i] = stateBuf[i * 4 + 2];
    return out;
  }

  function populationRate() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, stateReadFBO);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, stateBuf);
    let spikes = 0;
    for (let i = 0; i < W * H; i++) if (stateBuf[i * 4 + 3] > 0.5) spikes++;
    return spikes / (W * H);
  }

  function visualize(mode = 0) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(pVz);
    gl.uniform2f(u(pVz, "u_grid"), W, H);
    gl.uniform1f(u(pVz, "u_mode"), mode);
    bindTexUnits(gl, pVz, [
      { name: "u_state", tex: stateRead, unit: 0 },
      { name: "u_drive", tex: driveTex,  unit: 1 },
    ]);
    draw();
  }

  function reset() {
    const fill = (tex, data) => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, data);
    };
    fill(stateA, initState());
    fill(stateB, initState());
    fill(driveTex, zeros());
    // Re-roll the reservoir weights too
    const local = makeReservoirWeights(W, H, CFG.resSigmaLocal, CFG.resSparsity);
    const long  = makeReservoirWeights(W, H, CFG.resSigmaLong,  CFG.resSparsity);
    fill(w[0], local[0]); fill(w[1], local[1]); fill(w[2], local[2]);
    fill(w[3], long[0]);  fill(w[4], long[1]);  fill(w[5], long[2]);
  }

  return {
    gl, W, H,
    featureDim: W * H * 2,
    step, present, readTraces, populationRate,
    setDrive, visualize, reset,
  };
}
