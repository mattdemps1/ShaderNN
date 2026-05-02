#version 300 es
precision highp float;

in vec2 v_uv;
layout(location=0) out vec4 outColor;

uniform sampler2D u_state;
uniform sampler2D u_drive;
uniform float u_mode;   // 0 = membrane V, 1 = filtered firing rate (trace), 2 = input drive
uniform vec2  u_grid;

vec3 magma(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = vec3(0.0);
  c.r = smoothstep(0.0, 0.7, t);
  c.g = smoothstep(0.4, 1.0, t) * 0.85;
  c.b = smoothstep(0.0, 0.5, t) * (1.0 - smoothstep(0.5, 0.9, t));
  return c;
}

vec3 viridis(float t) {
  t = clamp(t, 0.0, 1.0);
  return vec3(
    smoothstep(0.0, 1.0, t) * 0.9,
    smoothstep(0.0, 1.0, sqrt(t)) * 0.85,
    smoothstep(0.6, 0.0, t) * 0.7 + 0.25
  );
}

void main() {
  vec2 uv = v_uv;

  if (u_mode > 1.5) {
    float d = texture(u_drive, uv).r;
    outColor = vec4(viridis(d * 1.2), 1.0);
    return;
  }

  vec4 s = texture(u_state, uv);
  float V     = s.r;
  float trace = s.b;
  float spike = s.a;

  if (u_mode > 0.5) {
    // Filtered firing rate — what the readout actually consumes
    outColor = vec4(magma(trace * 4.0), 1.0);
    return;
  }

  // Default: membrane voltage with spike highlights
  vec3 col = magma(trace * 0.8);
  col = mix(col, vec3(0.1, 0.2, 0.45) * clamp(V, 0.0, 1.0), 0.35);
  col += vec3(spike);
  outColor = vec4(col, 1.0);
}
