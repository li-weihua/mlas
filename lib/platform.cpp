/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    platform.cpp

Abstract:

    This module implements logic to select the best configuration for the
    this platform.

--*/

#include "mlasi.h"

#include <thread>
#include <mutex>

#if defined(MLAS_TARGET_POWER) && defined(__linux__)
#include <sys/auxv.h>
#endif

#if defined(MLAS_TARGET_ARM64)
#if defined(_WIN32)

// N.B. Support building with downlevel versions of the Windows SDK.
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
MLASCPUIDInfo::MLASCPUIDInfo() {
  has_arm_neon_dot_ = (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0);
}
#endif

#elif defined(__linux__)

#include <sys/auxv.h>
#include <asm/hwcap.h>
// N.B. Support building with older versions of asm/hwcap.h that do not define
// this capability bit.
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
MLASCPUIDInfo::MLASCPUIDInfo() { has_arm_neon_dot_ = ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0); }
#endif

#else

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
MLASCPUIDInfo::MLASCPUIDInfo() {}
#endif

#endif  // Windows vs Linux vs Unknown
#else   // not MLAS_TARGET_ARM64

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
MLASCPUIDInfo::MLASCPUIDInfo() {}
#endif

#endif  // MLAS_TARGET_ARM64

#ifdef MLAS_TARGET_AMD64_IX86

//
// Stores a vector to build a conditional load/store mask for vmaskmovps.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint32_t MlasMaskMoveAvx[8], 32) = {0, 1, 2, 3, 4, 5, 6, 7};

//
// Stores a table of AVX vmaskmovps/vmaskmovpd load/store masks.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const uint32_t MlasMaskMoveTableAvx[16], 32) = {
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0xFFFFFFFF,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
};

//
// Stores a table of AVX512 opmask register values.
//

MLAS_INTERNAL_DATA MLAS_DECLSPEC_ALIGN(const int16_t MlasOpmask16BitTableAvx512[16], 32) = {
    0x0000,
    0x0001,
    0x0003,
    0x0007,
    0x000F,
    0x001F,
    0x003F,
    0x007F,
    0x00FF,
    0x01FF,
    0x03FF,
    0x07FF,
    0x0FFF,
    0x1FFF,
    0x3FFF,
    0x7FFF,
};

//
// Reads the processor extended control register to determine platform
// capabilities.
//

#if !defined(_XCR_XFEATURE_ENABLED_MASK)
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

inline uint64_t
MlasReadExtendedControlRegister(
    unsigned int ext_ctrl_reg) {
#if defined(_WIN32)
  return _xgetbv(ext_ctrl_reg);
#else
  uint32_t eax, edx;

  __asm__(
      "xgetbv"
      : "=a"(eax), "=d"(edx)
      : "c"(ext_ctrl_reg));

  return ((uint64_t)edx << 32) | eax;
#endif
}

#endif  // MLAS_TARGET_AMD64_IX86

MLAS_PLATFORM::MLAS_PLATFORM(
    void)
/*++

Routine Description:

    This routine initializes the platform support for this library.

Arguments:

    None.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64_IX86)

  //
  // Default to the baseline SSE2 support.
  //

  this->GemmFloatKernel = MlasGemmFloatKernelSse;

#if defined(MLAS_TARGET_AMD64)

  this->ConvNchwFloatKernel = MlasConvNchwFloatKernelSse;
  this->NchwcBlockSize = 8;
  this->PreferredBufferAlignment = MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT;

  this->MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT;

#endif

  unsigned Cpuid1[4];
#if defined(_WIN32)
  __cpuid((int*)Cpuid1, 1);
#else
  __cpuid(1, Cpuid1[0], Cpuid1[1], Cpuid1[2], Cpuid1[3]);
#endif

  //
  // Check if the processor supports the AVX and OSXSAVE features.
  //

  if ((Cpuid1[2] & 0x18000000) == 0x18000000) {
    //
    // Check if the operating system supports saving SSE and AVX states.
    //

    uint64_t xcr0 = MlasReadExtendedControlRegister(_XCR_XFEATURE_ENABLED_MASK);

    if ((xcr0 & 0x6) == 0x6) {
      this->GemmFloatKernel = MlasGemmFloatKernelAvx;

#if defined(MLAS_TARGET_AMD64)

      this->KernelM1Routine = MlasSgemmKernelM1Avx;
      this->KernelM1TransposeBRoutine = MlasSgemmKernelM1TransposeBAvx;
      this->ConvNchwFloatKernel = MlasConvNchwFloatKernelAvx;

      //
      // Check if the processor supports AVX2/FMA3 features.
      //

      unsigned Cpuid7[4];
#if defined(_WIN32)
      __cpuidex((int*)Cpuid7, 7, 0);
#else
      __cpuid_count(7, 0, Cpuid7[0], Cpuid7[1], Cpuid7[2], Cpuid7[3]);
#endif

      if (((Cpuid1[2] & 0x1000) != 0) && ((Cpuid7[1] & 0x20) != 0)) {
        this->GemmFloatKernel = MlasGemmFloatKernelFma3;
        this->ConvNchwFloatKernel = MlasConvNchwFloatKernelFma3;

        //
        // Check if the processor supports Hybrid core architecture.
        //

        if ((Cpuid7[3] & 0x8000) != 0) {
          this->MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT * 4;
        }
      }

#endif  // MLAS_TARGET_AMD64
    }
  }

#endif  // MLAS_TARGET_AMD64_IX86
}

size_t
    MLASCALL
    MlasGetPreferredBufferAlignment(
        void)
/*++

Routine Description:

    This routine returns the preferred byte alignment for buffers that are used
    with this library. Buffers that are not byte aligned to this value will
    function, but will not achieve best performance.

Arguments:

    None.

Return Value:

    Returns the preferred byte alignment for buffers.

--*/
{
#if defined(MLAS_TARGET_AMD64)
  return GetMlasPlatform().PreferredBufferAlignment;
#else
  return MLAS_DEFAULT_PREFERRED_BUFFER_ALIGNMENT;
#endif
}

#ifdef MLAS_TARGET_AMD64_IX86

bool
    MLASCALL
    MlasPlatformU8S8Overflow(
        void) {
  const auto& p = GetMlasPlatform();
  return p.GemmU8U8Dispatch != p.GemmU8S8Dispatch;
}

#endif

thread_local size_t ThreadedBufSize = 0;
#ifdef _MSC_VER
thread_local std::unique_ptr<uint8_t, decltype(&_aligned_free)> ThreadedBufHolder(nullptr, &_aligned_free);
#else
thread_local std::unique_ptr<uint8_t, decltype(&free)> ThreadedBufHolder(nullptr, &free);
#endif
