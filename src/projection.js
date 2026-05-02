// Fixed random sparse projection from 28×28 input image (784 dims) to 16384
// reservoir-neuron drives. Each reservoir neuron picks `fanIn` random input
// pixels with random ±1 sign weights. Identical projection used for training
// and inference, so we generate it once at construction.

export function buildProjection(numNeurons, inputDim, fanIn) {
  // Per-neuron index list and weight list
  const indices = new Int32Array(numNeurons * fanIn);
  const weights = new Float32Array(numNeurons * fanIn);
  const norm    = 1.0 / Math.sqrt(fanIn);
  for (let n = 0; n < numNeurons; n++) {
    for (let k = 0; k < fanIn; k++) {
      indices[n * fanIn + k] = (Math.random() * inputDim) | 0;
      weights[n * fanIn + k] = (Math.random() < 0.5 ? -1 : 1) * norm;
    }
  }

  const driveBuf = new Float32Array(numNeurons);

  function project(image) {
    // image: Float32Array length inputDim, values in [0,1]
    for (let n = 0; n < numNeurons; n++) {
      let s = 0;
      const base = n * fanIn;
      for (let k = 0; k < fanIn; k++) {
        s += image[indices[base + k]] * weights[base + k];
      }
      // Half-wave rectify and scale to [0, ~1]: positive net input drives the neuron
      driveBuf[n] = Math.max(0, s);
    }
    return driveBuf;
  }

  return { project, fanIn };
}
