export const CFG = {
  gridW: 256,
  gridH: 256,
  numChannels: 16,
  hiddenDim: 96,
  perceptionDim: 48,   // 3 * numChannels
  stepsPerFrame: 2,
  updateProb: 0.5,
  aliveThreshold: 0.1,
  damageRadius: 20,
  targets: ["heart", "smiley", "lizard"],
  dataDir: "data",
};
