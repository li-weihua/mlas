#pragma once

#include <utility>

inline std::pair<float, float> get_min_max(float* input, int n) {
  float xmin = 1e20f;
  float xmax = -1e20f;

  for (int i = 0; i < n; ++i) {
    float x = input[i];

    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
  }

  return {xmin, xmax};
}

inline float get_max_diff(float* A, float* B, int n) {
  float diff = -0.0f;

  for (int i = 0; i < n; ++i) {
    float d = A[i] - B[i];
    if (d < 0) d = -d;

    if (d > diff) diff = d;
  }

  return diff;
}
