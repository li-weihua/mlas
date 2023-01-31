#pragma once

#include <cstddef>  // size_t
#include <cassert>

#include <iostream>

#include "../inc/mlas.h"

class Conv2d {
 public:
  Conv2d() {
    int in_channels = 48;
    int height = 2;
    int width = 40;

    int kernel_height = 2;
    int kernel_width = 3;

    int64_t InputShape[] = {int64_t(height), int64_t(width)};
    int64_t KernelShape[] = {int64_t(kernel_height), int64_t(kernel_width)};
    int64_t DilationShape[] = {int64_t(1), int64_t(1)};
    int64_t Padding[] = {int64_t(0), int64_t(1), int64_t(0), int64_t(1)};
    int64_t StrideShape[] = {int64_t(1), int64_t(1)};
    int64_t OutputShape[] = {int64_t(1), int64_t(width)};

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = MlasIdentityActivation;

    MlasConvPrepare(&params_,
                    2,
                    1,
                    1,
                    in_channels,
                    InputShape,
                    KernelShape,
                    DilationShape,
                    Padding,
                    StrideShape,
                    OutputShape,
                    in_channels,
                    &Activation,
                    &working_buffer_size_,
                    0.0f,
                    nullptr);

    std::cout << "workspace size: " << working_buffer_size_ << std::endl;
  }
  ~Conv2d() {}

  size_t GetBufferSize() {
    return working_buffer_size_;
  }

  void SetBuffer(const void* buffer) {
    buffer_ = reinterpret_cast<float*>(const_cast<void*>(buffer));
  }

  void DoForward(float* output, const float* input, const float* kernel, const float* bias) {
    MlasConv(&params_, input, kernel, bias, buffer_, output, nullptr);
  }

 private:
  MLAS_CONV_PARAMETERS params_;
  size_t working_buffer_size_ = 0;

  float* buffer_;
};
