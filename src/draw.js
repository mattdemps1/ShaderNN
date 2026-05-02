// Mouse/touch drawing onto a 28×28 logical grid, displayed on a larger canvas.
// Exposes `image` as a Float32Array(784) in [0,1] (top-left origin, row-major)
// — same convention as MNIST.
//
// We draw into a 140×140 supersampled buffer with anti-aliasing on, then
// area-average down to 28×28. This produces soft grey-edged strokes that
// match MNIST's appearance instead of the hard pixel blocks you get from
// drawing directly into a 28×28 buffer.

const N = 28;
const SUPER = 5;
const BN = N * SUPER;  // 140

export function buildDraw(canvas, opts = {}) {
  const onChange = opts.onChange || (() => {});
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;

  // High-res internal buffer where strokes are actually drawn
  const buffer = document.createElement("canvas");
  buffer.width = BN; buffer.height = BN;
  const bctx = buffer.getContext("2d");
  bctx.imageSmoothingEnabled = true;
  bctx.imageSmoothingQuality = "high";

  // 28×28 staging canvas used to area-average the buffer down
  const stage = document.createElement("canvas");
  stage.width = N; stage.height = N;
  const sctx = stage.getContext("2d");
  sctx.imageSmoothingEnabled = true;
  sctx.imageSmoothingQuality = "high";

  const image = new Float32Array(N * N);

  let drawing = false;
  let lastX = 0, lastY = 0;
  let dirty = true;

  function clientToBuffer(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = (e.clientX !== undefined ? e.clientX : e.touches[0].clientX) - rect.left;
    const cy = (e.clientY !== undefined ? e.clientY : e.touches[0].clientY) - rect.top;
    return {
      x: (cx / rect.width)  * BN,
      y: (cy / rect.height) * BN,
    };
  }

  function strokeTo(x, y) {
    bctx.strokeStyle = "#fff";
    bctx.lineCap = "round";
    bctx.lineJoin = "round";
    bctx.lineWidth = 12;  // ≈ 2.4 px in 28×28, MNIST-typical stroke width
    bctx.beginPath();
    bctx.moveTo(lastX, lastY);
    bctx.lineTo(x, y);
    bctx.stroke();
    lastX = x; lastY = y;
    dirty = true;
  }

  function readBuffer() {
    sctx.clearRect(0, 0, N, N);
    sctx.drawImage(buffer, 0, 0, N, N);
    const id = sctx.getImageData(0, 0, N, N);
    for (let i = 0; i < N * N; i++) {
      image[i] = id.data[i * 4] / 255;
    }
  }

  function render() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(60,60,80,0.18)";
    ctx.lineWidth = 1;
    const cell = canvas.width / N;
    ctx.beginPath();
    for (let i = 1; i < N; i++) {
      const p = Math.round(i * cell) + 0.5;
      ctx.moveTo(p, 0); ctx.lineTo(p, canvas.height);
      ctx.moveTo(0, p); ctx.lineTo(canvas.width, p);
    }
    ctx.stroke();
  }

  function tick() {
    if (dirty) {
      readBuffer();
      render();
      onChange(image);
      dirty = false;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  function clear() {
    bctx.fillStyle = "#000";
    bctx.fillRect(0, 0, BN, BN);
    image.fill(0);
    dirty = true;
  }
  clear();

  function pointerDown(e) {
    drawing = true;
    const { x, y } = clientToBuffer(e);
    lastX = x; lastY = y;
    strokeTo(x + 0.001, y + 0.001);
    e.preventDefault();
  }
  function pointerMove(e) {
    if (!drawing) return;
    const { x, y } = clientToBuffer(e);
    strokeTo(x, y);
    e.preventDefault();
  }
  function pointerUp() {
    drawing = false;
  }

  canvas.addEventListener("mousedown", pointerDown);
  canvas.addEventListener("mousemove", pointerMove);
  window.addEventListener("mouseup", pointerUp);
  canvas.addEventListener("touchstart", pointerDown);
  canvas.addEventListener("touchmove", pointerMove);
  canvas.addEventListener("touchend", pointerUp);

  return { image, clear, render, isEmpty: () => {
    for (let i = 0; i < image.length; i++) if (image[i] > 0.05) return false;
    return true;
  } };
}
