#version 300 es
precision highp float;

in vec2 v_uv;
layout(location=0) out vec4 outCurrent;

// Per-neuron input drive [0,1]: precomputed CPU-side as a fixed random sparse
// projection of the 28×28 input image. Stored in the .r channel of a 128×128
// R32F texture (uploaded as RGBA, only .r used).
uniform sampler2D u_drive;

uniform vec2  u_grid;
uniform float u_driveScale;   // multiplier: drive*driveScale = Poisson rate
uniform float u_noiseRate;
uniform float u_time;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vec2 px = floor(v_uv * u_grid);

  float current = 0.0;

  // Background drizzle so the reservoir doesn't go silent on blank input.
  float n = hash(px + vec2(u_time * 7.13, u_time * 3.71));
  if (n < u_noiseRate) current += 0.5;

  // Input-driven Poisson spikes.
  float drive = texture(u_drive, v_uv).r;
  float rate = clamp(drive * u_driveScale, 0.0, 0.95);
  if (rate > 0.0) {
    float s = hash(px + vec2(u_time * 11.7, u_time * 5.3));
    if (s < rate) current += 1.2;
  }

  outCurrent = vec4(current, 0.0, 0.0, 0.0);
}
