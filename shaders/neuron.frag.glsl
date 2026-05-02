#version 300 es
precision highp float;

in vec2 v_uv;
layout(location=0) out vec4 outState;

uniform sampler2D u_state;
uniform sampler2D u_input;
// local 3×3 stride-1 reservoir weights (signed, fixed)
uniform sampler2D u_w0;
uniform sampler2D u_w1;
uniform sampler2D u_w2;
// long-range 3×3 stride-N reservoir weights (signed, fixed)
uniform sampler2D u_w3;
uniform sampler2D u_w4;
uniform sampler2D u_w5;

uniform vec2  u_grid;
uniform float u_leak;
uniform float u_vThr;
uniform float u_vReset;
uniform float u_tRef;
uniform float u_traceDecay;
uniform float u_thrJitter;
uniform float u_vNoise;
uniform float u_longStride;
uniform float u_time;

float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec4 sampleState(vec2 px) {
  vec2 uv = (px + 0.5) / u_grid;
  return texture(u_state, fract(uv));
}

void main() {
  vec2 px = floor(v_uv * u_grid);

  vec4 w0 = texture(u_w0, v_uv);
  vec4 w1 = texture(u_w1, v_uv);
  vec4 w2 = texture(u_w2, v_uv);
  float w[9];
  w[0]=w0.r; w[1]=w0.g; w[2]=w0.b; w[3]=w0.a;
  w[4]=w1.r; w[5]=w1.g; w[6]=w1.b; w[7]=w1.a;
  w[8]=w2.r;

  vec4 lw0 = texture(u_w3, v_uv);
  vec4 lw1 = texture(u_w4, v_uv);
  vec4 lw2 = texture(u_w5, v_uv);
  float cw[9];
  cw[0]=lw0.r; cw[1]=lw0.g; cw[2]=lw0.b; cw[3]=lw0.a;
  cw[4]=lw1.r; cw[5]=lw1.g; cw[6]=lw1.b; cw[7]=lw1.a;
  cw[8]=lw2.r;

  vec2 OFF[9] = vec2[9](
    vec2(-1.,-1.), vec2(0.,-1.), vec2(1.,-1.),
    vec2(-1., 0.), vec2(0., 0.), vec2(1., 0.),
    vec2(-1., 1.), vec2(0., 1.), vec2(1., 1.)
  );

  float s = u_longStride;
  vec2 OFF_LONG[9] = vec2[9](
    vec2(-1.,-s), vec2(0.,-s), vec2(1.,-s),
    vec2(-1., 0.), vec2(0., 0.), vec2(1., 0.),
    vec2(-1., s), vec2(0., s), vec2(1., s)
  );

  float syn = 0.0;
  for (int k = 0; k < 9; ++k) {
    syn += sampleState(px + OFF[k]).a * w[k];
  }
  for (int k = 0; k < 9; ++k) {
    syn += sampleState(px + OFF_LONG[k]).a * cw[k];
  }

  vec4 self = sampleState(px);
  float V    = self.r;
  float refr = self.g;
  float trace= self.b;

  float inj = texture(u_input, v_uv).r;

  float canInt = (refr > 0.5) ? 0.0 : 1.0;
  V = mix(V, V * u_leak + syn + inj, canInt);

  float vn = (hash(px + vec2(u_time * 17.13, u_time * 23.71)) - 0.5) * 2.0 * u_vNoise;
  V += vn * canInt;

  float thrOff = (hash(px + vec2(13.7, 91.3)) - 0.5) * u_thrJitter;
  float thr = u_vThr + thrOff;

  float spike = (V >= thr) ? 1.0 : 0.0;
  V = mix(V, u_vReset, spike);
  refr = max(refr - 1.0, 0.0);
  refr = mix(refr, u_tRef, spike);

  trace = trace * u_traceDecay + spike * (1.0 - u_traceDecay);

  outState = vec4(V, refr, trace, spike);
}
