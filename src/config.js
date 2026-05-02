export const CFG = {
  // Reservoir grid (one LIF neuron per pixel)
  grid: 128,             // → 16384 neurons
  numClasses: 10,

  // LIF dynamics
  vThr: 0.55,
  vReset: 0.0,
  leak: 0.92,
  tRef: 3,
  thrJitter: 0.10,
  vNoise: 0.02,
  traceDecay: 0.94,      // low-pass on spikes; trace = readout features

  // Reservoir connectivity (FIXED — drawn once at init, never updated)
  // Local 3×3 stride-1 + long-range 3×3 stride-16. Signed weights so the
  // network has both excitatory and inhibitory connections without needing
  // hardcoded global inhibition.
  longStride: 16,
  resSigmaLocal: 0.18,   // std-dev of local weights
  resSigmaLong:  0.12,   // std-dev of long-range weights
  resSparsity:   0.55,   // fraction of slots zeroed (echo-state friendly)

  // Input projection (fixed random sparse 784 → 16384)
  inputFanIn: 8,         // each reservoir neuron gets input from this many random pixels
  inputDriveScale: 1.4,  // multiplier on Poisson rate fed to shader
  noiseRate: 0.002,      // tiny background drizzle to keep neurons alive

  // Simulation cadence
  substeps: 4,           // GPU steps per requestAnimationFrame (during training: per example)
  presentSteps: 48,      // reservoir steps per MNIST example
  midSnapshotStep: 24,   // grab a second trace snapshot at this step → 2× features

  // Readout (linear softmax classifier, trained on CPU)
  readoutLR: 0.05,
  readoutL2: 1e-5,
  readoutMomentum: 0.9,

  // Training schedule
  trainBatchPerFrame: 1, // examples processed per animation frame
  valEvery: 50,          // evaluate val accuracy every N training examples
  valBatch: 64,          // # val examples per evaluation
  accEMA: 0.02,          // EMA factor for live training accuracy display
};
