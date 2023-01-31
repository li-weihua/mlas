/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    activate.cpp

Abstract:

    This module implements the fused activation and bias addition routines.

--*/

#include "mlasi.h"

//
// Templates for bias addition functions.
//

template <bool AddBias>
struct MLAS_BIAS_ADDITION;

template <>
struct MLAS_BIAS_ADDITION<true> {
  MLAS_FLOAT32X4 BiasBroadcast;

  void LoadNext(const float*& Bias) {
    BiasBroadcast = MlasBroadcastFloat32x4(Bias++);
  }

  MLAS_FLOAT32X4 Add(MLAS_FLOAT32X4 Value) {
    return MlasAddFloat32x4(Value, BiasBroadcast);
  }

  float Add(float Value) {
    return Value + MlasExtractLaneFloat32x4<0>(BiasBroadcast);
  }
};

template <>
struct MLAS_BIAS_ADDITION<false> {
  void LoadNext(const float*& Bias) {
    MLAS_UNREFERENCED_PARAMETER(Bias);
  }

  MLAS_FLOAT32X4 Add(MLAS_FLOAT32X4 Value) {
    return Value;
  }

  float Add(float Value) {
    return Value;
  }
};

//
// Templates for activation functions.
//

template <MLAS_ACTIVATION_KIND ActivationKind>
struct MLAS_ACTIVATION_FUNCTION;

template <>
struct MLAS_ACTIVATION_FUNCTION<MlasIdentityActivation> {
  MLAS_ACTIVATION_FUNCTION(const MLAS_ACTIVATION* Activation) {
    MLAS_UNREFERENCED_PARAMETER(Activation);
  }

  MLAS_FLOAT32X4 Activate(MLAS_FLOAT32X4 Value) {
    return Value;
  }

  float Activate(float Value) {
    return Value;
  }
};

template <MLAS_ACTIVATION_KIND ActivationKind, bool AddBias>
void MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc)
/*++

Routine Description:

    This routine steps over the output matrix and invokes the templated bias
    addition and activation functions.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
  MLAS_ACTIVATION_FUNCTION<ActivationKind> ActivationFunction(Activation);
  MLAS_BIAS_ADDITION<AddBias> BiasAddition;

  //
  // Step through each row of the output matrix.
  //

  while (M-- > 0) {
    float* buffer = Buffer;
    size_t n = N;

    BiasAddition.LoadNext(Bias);

    if (n >= 4) {
      do {
        MLAS_FLOAT32X4 Vector = BiasAddition.Add(MlasLoadFloat32x4(buffer));
        MlasStoreFloat32x4(buffer, ActivationFunction.Activate(Vector));
        buffer += 4;
        n -= 4;

      } while (n >= 4);
    }

    while (n > 0) {
      float Scalar = BiasAddition.Add(*buffer);
      *buffer++ = ActivationFunction.Activate(Scalar);
      n -= 1;
    }

    Buffer += ldc;
  }
}

template <>
inline void
MlasActivationKernel<MlasIdentityActivation, false>(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc)
/*++

Routine Description:

    This routine is invoked for the special case of an identity operation with
    no bias addition, which translates to a no-op.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
  //
  // No operation.
  //

  MLAS_UNREFERENCED_PARAMETER(Activation);
  MLAS_UNREFERENCED_PARAMETER(Buffer);
  MLAS_UNREFERENCED_PARAMETER(Bias);
  MLAS_UNREFERENCED_PARAMETER(M);
  MLAS_UNREFERENCED_PARAMETER(N);
  MLAS_UNREFERENCED_PARAMETER(ldc);
}

template <MLAS_ACTIVATION_KIND ActivationKind>
inline void
MlasActivationKernel(
    const MLAS_ACTIVATION* Activation,
    float* Buffer,
    const float* Bias,
    size_t M,
    size_t N,
    size_t ldc)
/*++

Routine Description:

    This routine invokes the appropriate activation kernel based on the
    optional bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
  if (Bias != nullptr) {
    MlasActivationKernel<ActivationKind, true>(Activation, Buffer, Bias, M, N, ldc);
  } else {
    MlasActivationKernel<ActivationKind, false>(Activation, Buffer, Bias, M, N, ldc);
  }
}

void
    MLASCALL
    MlasActivation(
        const MLAS_ACTIVATION* Activation,
        float* Buffer,
        const float* Bias,
        size_t M,
        size_t N,
        size_t ldc)
/*++

Routine Description:

    This routine applies an activation function to the output matrix after
    optionally adding a bias vector.

Arguments:

    Activation - Supplies the parameters for the activation.

    Buffer - Supplies the output matrix.

    Bias - Supplies the optional bias vector.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    ldc - Supplies the number of elements per row of the output matrix.

Return Value:

    None.

--*/
{
  switch (Activation->ActivationKind) {
    case MlasIdentityActivation: {
      MlasActivationKernel<MlasIdentityActivation>(Activation, Buffer, Bias, M, N, ldc);
      break;
    }
  }
}
