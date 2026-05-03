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
  traceDecay: 0.85,      // low-pass on spikes; trace = readout features.
                         // Half-life ~4 steps so trace builds within presentSteps.

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
  inputDriveScale: 2.0,  // multiplier on Poisson rate fed to shader
  noiseRate: 0.001,      // tiny background drizzle to keep neurons alive

  // Simulation cadence
  substeps: 4,           // GPU steps per requestAnimationFrame (during training: per example)
  presentSteps: 24,      // reservoir steps per MNIST example
  midSnapshotStep: 12,   // grab a second trace snapshot at this step → 2× features

  // Readout (one-hidden-layer MLP, trained on CPU with Adam).
  // Each feature block (reservoir trace, raw pixels) is L2-normalized
  // independently in main.featurize() so both contribute meaningfully.
  hiddenDim: 64,         // ReLU hidden units between features and softmax
  readoutLR: 1e-3,
  readoutL2: 1e-4,       // decoupled (AdamW) weight decay
  adamBeta1: 0.9,
  adamBeta2: 0.999,
  adamEps: 1e-8,
  // Focal-loss exponent: gradient scaled by (1 - p_correct)^gamma. Off:
  // it starves learning once p_correct rises and isn't useful for balanced MNIST.
  focalGamma: 0,

  // Training schedule
  trainBatchPerFrame: 4, // examples processed per animation frame
  valEvery: 50,          // evaluate val accuracy every N training examples
  valBatch: 64,          // # val examples per evaluation
  accEMA: 0.02,          // EMA factor for live training accuracy display
};
