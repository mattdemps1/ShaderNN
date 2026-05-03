import { augment } from "./augment.js";

// Loads data/mnist_subset.bin produced by build_mnist.py.
//
// File layout (little-endian):
//   uint32 magic = 0x4D4E4953
//   uint32 numTrain
//   uint32 numTest
//   uint32 imageSize = 784
//   uint8  trainImages[numTrain * 784]
//   uint8  trainLabels[numTrain]
//   uint8  testImages[numTest * 784]
//   uint8  testLabels[numTest]

const MAGIC = 0x4D4E4953;

export async function loadMNIST(url = "./data/mnist_subset.bin") {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  if (magic !== MAGIC) throw new Error(`bad magic 0x${magic.toString(16)} in ${url}`);
  const numTrain = dv.getUint32(4, true);
  const numTest  = dv.getUint32(8, true);
  const imgSize  = dv.getUint32(12, true);
  if (imgSize !== 784) throw new Error(`unexpected image size ${imgSize}`);

  let off = 16;
  const trainImages = new Uint8Array(buf, off, numTrain * 784);
  off += numTrain * 784;
  const trainLabels = new Uint8Array(buf, off, numTrain);
  off += numTrain;
  const testImages  = new Uint8Array(buf, off, numTest * 784);
  off += numTest * 784;
  const testLabels  = new Uint8Array(buf, off, numTest);

  // Pre-decode to Float32 (one big array each, normalized to [0,1])
  const trainF = new Float32Array(numTrain * 784);
  for (let i = 0; i < trainImages.length; i++) trainF[i] = trainImages[i] / 255;
  const testF = new Float32Array(numTest * 784);
  for (let i = 0; i < testImages.length; i++) testF[i] = testImages[i] / 255;

  function getImage(images, idx) {
    return images.subarray(idx * 784, (idx + 1) * 784);
  }

  // Shuffled train iterator that loops forever
  let order = shuffledIndices(numTrain);
  let cursor = 0;
  let useAugment = true;
  function nextTrain() {
    if (cursor >= order.length) {
      order = shuffledIndices(numTrain);
      cursor = 0;
    }
    const i = order[cursor++];
    const raw = getImage(trainF, i);
    return { image: useAugment ? augment(raw) : raw, label: trainLabels[i], index: i };
  }

  function getTest(idx) {
    return { image: getImage(testF, idx), label: testLabels[idx], index: idx };
  }

  return {
    numTrain, numTest,
    nextTrain, getTest,
    setAugment(on) { useAugment = !!on; },
    trainImages: trainF, trainLabels,
    testImages:  testF,  testLabels,
  };
}

function shuffledIndices(n) {
  const arr = new Int32Array(n);
  for (let i = 0; i < n; i++) arr[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
  return arr;
}
