import { CFG } from "./config.js";
import { getGL, program, quad, makeFloatTex, makeFBO } from "./glutil.js";

const W = CFG.gridW, H = CFG.gridH;

const VERT = `#version 300 es
layout(location=0) in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Display shader: renders texture 0 RGB. Alpha encodes cell alive state;
// dead cells show as a dark background so you can see the grid boundary.
const DISPLAY_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D uState0;
void main() {
  vec4 s = texture(uState0, v_uv);
  vec3 rgb = clamp(s.rgb, 0.0, 1.0);
  float a   = clamp(s.a,   0.0, 1.0);
  fragColor = vec4(mix(vec3(0.04, 0.04, 0.06), rgb, a), 1.0);
}`;

// Grid width/height constants embedded for the damage radius calculation.
const GW = CFG.gridW.toFixed(1), GH = CFG.gridH.toFixed(1);

// Main CA update shader.
// Channels 0-15 packed as 4 RGBA32F textures (4 channels each).
// Perception = [identity(16), Sobel_x(16), Sobel_y(16)] = 48 values = 12 vec4s.
// MLP: fc1(48→96, ReLU) → fc2(96→16).
// UBO holds weights packed as vec4 arrays to avoid std140 float padding.
const CA_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;

layout(location=0) out vec4 out0;
layout(location=1) out vec4 out1;
layout(location=2) out vec4 out2;
layout(location=3) out vec4 out3;

uniform sampler2D uState0, uState1, uState2, uState3;
uniform vec2  uTexelSize;
uniform float uRandSeed;
uniform vec2  uDamageCenter; // normalised UV, (-1,-1) = inactive
uniform float uDamageRadius; // radius in pixels

// std140 vec4 packing avoids the 16-byte-per-float padding in float arrays.
// fc1: 96 rows × 48 cols  → 96*48/4 = 1152 vec4s  (18432 bytes)
// b1:  96 biases          → 96/4   = 24  vec4s  (  384 bytes)
// fc2: 16 rows × 96 cols  → 16*96/4 = 384 vec4s  ( 6144 bytes)
// b2:  16 biases          → 16/4   = 4   vec4s  (   64 bytes)
//                                      Total   = 25024 bytes
layout(std140) uniform Weights {
  vec4 fc1[1152];
  vec4 b1[24];
  vec4 fc2[384];
  vec4 b2[4];
} W;

// Wang hash → uniform float in [0,1] without a random texture.
float wangHash(vec2 uv, float seed) {
  uvec2 p = uvec2(uv * 65535.0);
  uint  n = p.x * 1664525u + p.y * 1013904223u + uint(seed * 4096.0);
  n = (n ^ (n >> 16u)) * 0x45d9f3bu;
  n = (n ^ (n >> 16u)) * 0x45d9f3bu;
  n =  n ^ (n >> 16u);
  return float(n) / 4294967295.0;
}

void fetchCell(vec2 uv, out vec4 s0, out vec4 s1, out vec4 s2, out vec4 s3) {
  s0 = texture(uState0, uv);
  s1 = texture(uState1, uv);
  s2 = texture(uState2, uv);
  s3 = texture(uState3, uv);
}

// Row j of fc1 (48 cols = 12 vec4-blocks) → block k.
vec4 fc1w(int j, int k) { return W.fc1[j * 12 + k]; }
// Row j of fc2 (96 cols = 24 vec4-blocks) → block k.
vec4 fc2w(int j, int k) { return W.fc2[j * 24 + k]; }

void main() {
  vec2 dx = vec2(uTexelSize.x, 0.0);
  vec2 dy = vec2(0.0, uTexelSize.y);

  // ── 1. 3×3 neighbourhood (36 texture fetches) ──────────────────────────
  vec4 n00_0,n00_1,n00_2,n00_3; fetchCell(v_uv-dx-dy, n00_0,n00_1,n00_2,n00_3);
  vec4 n10_0,n10_1,n10_2,n10_3; fetchCell(v_uv   -dy, n10_0,n10_1,n10_2,n10_3);
  vec4 n20_0,n20_1,n20_2,n20_3; fetchCell(v_uv+dx-dy, n20_0,n20_1,n20_2,n20_3);
  vec4 n01_0,n01_1,n01_2,n01_3; fetchCell(v_uv-dx,    n01_0,n01_1,n01_2,n01_3);
  vec4 c0,c1,c2,c3;             fetchCell(v_uv,        c0,c1,c2,c3);
  vec4 n21_0,n21_1,n21_2,n21_3; fetchCell(v_uv+dx,    n21_0,n21_1,n21_2,n21_3);
  vec4 n02_0,n02_1,n02_2,n02_3; fetchCell(v_uv-dx+dy, n02_0,n02_1,n02_2,n02_3);
  vec4 n12_0,n12_1,n12_2,n12_3; fetchCell(v_uv   +dy, n12_0,n12_1,n12_2,n12_3);
  vec4 n22_0,n22_1,n22_2,n22_3; fetchCell(v_uv+dx+dy, n22_0,n22_1,n22_2,n22_3);

  // ── 2. Alive mask (alpha = channel 3 of texture 0) ──────────────────────
  float maxA = max(max(max(n00_0.a,n10_0.a),max(n20_0.a,n01_0.a)),
                   max(max(c0.a,   n21_0.a),max(max(n02_0.a,n12_0.a),n22_0.a)));
  float alive = step(${CFG.aliveThreshold.toFixed(2)}, maxA);

  // ── 3. Perception: identity + Sobel_x + Sobel_y (12 vec4s) ─────────────
  // Kx=[[-1,0,1],[-2,0,2],[-1,0,1]]/8, Ky=Kx^T, applied to each tex vec4.
  vec4 perc[12];
  perc[0]=c0; perc[1]=c1; perc[2]=c2; perc[3]=c3;
  perc[4] =((-n00_0-2.0*n01_0-n02_0)+(n20_0+2.0*n21_0+n22_0))/8.0;
  perc[5] =((-n00_1-2.0*n01_1-n02_1)+(n20_1+2.0*n21_1+n22_1))/8.0;
  perc[6] =((-n00_2-2.0*n01_2-n02_2)+(n20_2+2.0*n21_2+n22_2))/8.0;
  perc[7] =((-n00_3-2.0*n01_3-n02_3)+(n20_3+2.0*n21_3+n22_3))/8.0;
  perc[8] =((-n00_0-2.0*n10_0-n20_0)+(n02_0+2.0*n12_0+n22_0))/8.0;
  perc[9] =((-n00_1-2.0*n10_1-n20_1)+(n02_1+2.0*n12_1+n22_1))/8.0;
  perc[10]=((-n00_2-2.0*n10_2-n20_2)+(n02_2+2.0*n12_2+n22_2))/8.0;
  perc[11]=((-n00_3-2.0*n10_3-n20_3)+(n02_3+2.0*n12_3+n22_3))/8.0;

  // ── 4. fc1: 48 → 96, ReLU ────────────────────────────────────────────────
  vec4 hidden[24];
  for (int j = 0; j < 24; j++) {
    vec4 acc = vec4(0.0);
    for (int k = 0; k < 12; k++) {
      acc.x += dot(fc1w(j*4+0, k), perc[k]);
      acc.y += dot(fc1w(j*4+1, k), perc[k]);
      acc.z += dot(fc1w(j*4+2, k), perc[k]);
      acc.w += dot(fc1w(j*4+3, k), perc[k]);
    }
    hidden[j] = max(vec4(0.0), acc + W.b1[j]);
  }

  // ── 5. fc2: 96 → 16 ──────────────────────────────────────────────────────
  vec4 delta[4];
  for (int j = 0; j < 4; j++) {
    vec4 acc = vec4(0.0);
    for (int k = 0; k < 24; k++) {
      acc.x += dot(fc2w(j*4+0, k), hidden[k]);
      acc.y += dot(fc2w(j*4+1, k), hidden[k]);
      acc.z += dot(fc2w(j*4+2, k), hidden[k]);
      acc.w += dot(fc2w(j*4+3, k), hidden[k]);
    }
    delta[j] = acc + W.b2[j];
  }

  // ── 6. Stochastic mask ────────────────────────────────────────────────────
  float stoch = step(0.5, wangHash(v_uv, uRandSeed));

  // ── 7. Damage mask ────────────────────────────────────────────────────────
  float dmg = 1.0;
  if (uDamageCenter.x >= 0.0) {
    // Convert UV difference to pixel distance for an isotropic brush.
    float dist = length((v_uv - uDamageCenter) * vec2(${GW}, ${GH}));
    dmg = step(uDamageRadius, dist);
  }

  // ── 8. Update ─────────────────────────────────────────────────────────────
  // alive * stoch: only alive cells receive the delta (matches Python).
  // * alive on the full state: dead cells are zeroed, matching Python's
  //   alive_post mask (approximated with alive_pre for simplicity).
  // * dmg: damage brush zeroes cells.
  // clamp(-1,1) matches Python ca_step; without it hidden channels drift
  // unboundedly over thousands of browser steps causing white chaos.
  float delta_mask = alive * stoch;
  out0 = clamp((c0 + delta[0] * delta_mask) * alive * dmg, -1.0, 1.0);
  out1 = clamp((c1 + delta[1] * delta_mask) * alive * dmg, -1.0, 1.0);
  out2 = clamp((c2 + delta[2] * delta_mask) * alive * dmg, -1.0, 1.0);
  out3 = clamp((c3 + delta[3] * delta_mask) * alive * dmg, -1.0, 1.0);
}`;

export function buildCA(canvas) {
  const gl = getGL(canvas);

  const maxDraw = gl.getParameter(gl.MAX_DRAW_BUFFERS);
  if (maxDraw < 4) throw new Error(`MAX_DRAW_BUFFERS=${maxDraw}, need ≥4`);

  const vao          = quad(gl);
  const caProgram    = program(gl, VERT, CA_FRAG,      "ca");
  const displayProg  = program(gl, VERT, DISPLAY_FRAG, "display");

  // 2 ping-pong sets × 4 RGBA32F textures each
  const readTex  = Array.from({ length: 4 }, () => makeFloatTex(gl, W, H));
  const writeTex = Array.from({ length: 4 }, () => makeFloatTex(gl, W, H));
  const readFBO  = makeFBO(gl, readTex);
  const writeFBO = makeFBO(gl, writeTex);

  const uLoc = {
    texelSize:   gl.getUniformLocation(caProgram, "uTexelSize"),
    randSeed:    gl.getUniformLocation(caProgram, "uRandSeed"),
    dmgCenter:   gl.getUniformLocation(caProgram, "uDamageCenter"),
    dmgRadius:   gl.getUniformLocation(caProgram, "uDamageRadius"),
    states:      [0,1,2,3].map(i => gl.getUniformLocation(caProgram, `uState${i}`)),
    displayState: gl.getUniformLocation(displayProg, "uState0"),
  };

  let dmgCenter = [-1, -1];
  let frameIdx  = 0;

  function reattach(fbo, texSet) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    texSet.forEach((t, i) =>
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
  }

  function step() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, writeFBO);
    gl.viewport(0, 0, W, H);
    gl.useProgram(caProgram);
    gl.bindVertexArray(vao);

    gl.uniform2f(uLoc.texelSize, 1 / W, 1 / H);
    gl.uniform1f(uLoc.randSeed,  (frameIdx++ * 1.6180339887) % 1024);
    gl.uniform2f(uLoc.dmgCenter, dmgCenter[0], dmgCenter[1]);
    gl.uniform1f(uLoc.dmgRadius, CFG.damageRadius);

    readTex.forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(uLoc.states[i], i);
    });

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Swap tex references then re-attach to their FBOs
    for (let i = 0; i < 4; i++) {
      const tmp = readTex[i]; readTex[i] = writeTex[i]; writeTex[i] = tmp;
    }
    reattach(readFBO,  readTex);
    reattach(writeFBO, writeTex);
  }

  function display() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(displayProg);
    gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, readTex[0]);
    gl.uniform1i(uLoc.displayState, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function seed(rgba) {
    const cx = Math.floor(W / 2), cy = Math.floor(H / 2);
    const px = new Float32Array([rgba[0], rgba[1], rgba[2], rgba[3]]);
    gl.bindTexture(gl.TEXTURE_2D, readTex[0]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, cx, cy, 1, 1, gl.RGBA, gl.FLOAT, px);
  }

  function reset(rgba) {
    const zeros = new Float32Array(W * H * 4);
    readTex.forEach(t => {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, zeros);
    });
    frameIdx = 0;
    if (rgba) seed(rgba);
  }

  function setDamage(uvX, uvY) { dmgCenter = [uvX, uvY]; }
  function clearDamage()        { dmgCenter = [-1, -1]; }

  function bindWeights(ubo) {
    const idx = gl.getUniformBlockIndex(caProgram, "Weights");
    if (idx === gl.INVALID_INDEX) throw new Error("Weights UBO block not found");
    gl.uniformBlockBinding(caProgram, idx, 0);
    gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ubo);
  }

  return { gl, caProgram, step, display, seed, reset, setDamage, clearDamage, bindWeights };
}
