#include <iostream>
#include <cstdlib>

#include "../inc/mlas.h"

std::pair<float, float> get_min_max(float* input, int n) {
  float xmin = 1e20f;
  float xmax = -1e20f;

  for (int i = 0; i < n; ++i) {
    float x = input[i];

    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
  }

  return {xmin, xmax};
}

float get_max_diff(float* A, float* B, int n) {
  float diff = -0.0f;

  for (int i = 0; i < n; ++i) {
    float d = A[i] - B[i];
    if (d < 0) d = -d;

    if (d > diff) diff = d;
  }

  return diff;
}

// C = A * B
void sgemm_ref(float* A, float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int p = 0; p < k; ++p) {
        sum += A[i * k + p] * B[p * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

int main() {
  int m = 100;
  int n = 100;
  int k = 100;

  float* A = (float*)std::malloc(sizeof(float) * m * k);
  float* B = (float*)std::malloc(sizeof(float) * k * n);

  float* C0 = (float*)std::malloc(sizeof(float) * m * n);
  float* C1 = (float*)std::malloc(sizeof(float) * m * n);

  // init A and B
  for (int i = 0; i < m * k; ++i) A[i] = (float)i / (m * k);
  for (int i = 0; i < k * n; ++i) B[i] = (float)(i % k) / k;

  sgemm_ref(A, B, C0, m, n, k);
  MlasGemm(CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0f, C1, n, nullptr);

  auto r0 = get_min_max(C0, m * n);
  auto r1 = get_min_max(C1, m * n);
  auto diff = get_max_diff(C0, C1, m * n);

  std::cout << r0.first << ", " << r0.second << std::endl;
  std::cout << r1.first << ", " << r0.second << std::endl;
  std::cout << diff << std::endl;

  // free memory
  free(A);
  free(B);
  free(C0);
  free(C1);

  return 0;
}
