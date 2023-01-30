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

#if defined(__linux__)
#include <sys/syscall.h>
#endif

bool MlasInitAMX() {
  /*
#if defined(__linux__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

    unsigned long bitmask = 0;
    long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (rc) {
        return false;
    }
    rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (rc) {
        return false;
    }
    if (bitmask & XFEATURE_MASK_XTILE) {
        return true;
    }
    return false;
#else
    return true;
#endif
*/
  return false;
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

  this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Sse;
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

#if defined(_MSC_VER)

  //
  // Check if the processor supports SSE 4.1 instructions.
  //

  if ((Cpuid1[2] & 0x80000) != 0) {
    this->GemmU8S8Dispatch = &MlasGemmU8S8DispatchSse41;
  }

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
      this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Avx;

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

        //
        // Check if the processor supports Hybrid core architecture.
        //

        if ((Cpuid7[3] & 0x8000) != 0) {
          this->MaximumThreadCount = MLAS_MAXIMUM_THREAD_COUNT * 4;
        }

        //
        // Check if the processor supports AVXVNNI features.
        //

        unsigned Cpuid7_1[4];
#if defined(_WIN32)
        __cpuidex((int*)Cpuid7_1, 7, 1);
#else
        __cpuid_count(7, 1, Cpuid7_1[0], Cpuid7_1[1], Cpuid7_1[2], Cpuid7_1[3]);
#endif

        if ((Cpuid7_1[0] & 0x10) != 0) {
        }

#if !defined(ORT_MINIMAL_BUILD)

        //
        // Check if the processor supports AVX512F features and the
        // operating system supports saving AVX512F state.
        //

        if (((Cpuid7[1] & 0x10000) != 0) && ((xcr0 & 0xE0) == 0xE0)) {
          //
          // Check if the processor supports AVX512 core features
          // (AVX512BW/AVX512DQ/AVX512VL).
          //

          if ((Cpuid7[1] & 0xC0020000) == 0xC0020000) {
            //
            // Check if the processor supports AVX512VNNI.
            //

            if ((Cpuid7[2] & 0x800) != 0) {
            }
          }
        }

#ifdef MLAS_AMX_SUPPORTED
        //
        // Check if the processor supports AMX-TILE and AMX-INT8
        // features.
        //
        if ((Cpuid7[3] & 0b1 << 24) != 0 && (Cpuid7[3] & 0b1 << 25) != 0) {
          if (MlasInitAMX()) {
            this->GemmU8U8Dispatch = &MlasGemmU8S8DispatchAmx;
            this->GemmU8S8Dispatch = &MlasGemmU8S8DispatchAmx;
          }
        }
#endif  // MLAS_AMX_SUPPORTED

#endif  // ORT_MINIMAL_BUILD
      }

#endif  // MLAS_TARGET_AMD64
    }
  }

#endif  // MLAS_TARGET_AMD64_IX86

#if defined(MLAS_TARGET_ARM64)

  this->GemmU8X8Dispatch = &MlasGemmU8X8DispatchNeon;
  this->SymmQgemmDispatch = &MlasSymmQgemmS8DispatchNeon;
  this->ConvSymU8S8Dispatch = &MlasConvSymU8DispatchNeon;
  this->ConvSymS8S8Dispatch = &MlasConvSymS8DispatchNeon;

  //
  // Check if the processor supports ASIMD dot product instructions.
  //

  bool HasDotProductInstructions;

#if defined(_WIN32)
  HasDotProductInstructions = (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0);
#elif defined(__linux__)
  HasDotProductInstructions = MLAS_CPUIDINFO::GetCPUIDInfo().HasArmNeonDot();
#else
  HasDotProductInstructions = false;
#endif

  if (HasDotProductInstructions) {
    this->GemmU8X8Dispatch = &MlasGemmU8X8DispatchUdot;
    this->SymmQgemmDispatch = &MlasSymmQgemmS8DispatchSdot;
    this->ConvSymU8S8Dispatch = &MlasConvSymU8DispatchDot;
    this->ConvSymS8S8Dispatch = &MlasConvSymS8DispatchDot;
  }

#endif  // MLAS_TARGET_ARM64
#if defined(MLAS_TARGET_POWER)
  this->GemmFloatKernel = MlasSgemmKernel;
  this->GemmDoubleKernel = MlasDgemmKernel;
  this->QuantizeLinearS8Kernel = MlasQuantizeLinearS8Kernel;
  this->QuantizeLinearU8Kernel = MlasQuantizeLinearU8Kernel;

#if defined(__linux__)
  unsigned long hwcap2 = getauxval(AT_HWCAP2);

  bool HasP9Instructions = hwcap2 & PPC_FEATURE2_ARCH_3_00;
  if (HasP9Instructions) {
    this->QuantizeLinearS8Kernel = MlasQuantizeLinearS8KernelVSX;
    this->QuantizeLinearU8Kernel = MlasQuantizeLinearU8KernelVSX;
  }

#if defined(POWER10)
#if (defined(__GNUC__) && ((__GNUC__ > 10) || (__GNUC__ == 10 && __GNUC_MINOR__ >= 2))) || \
    (defined(__clang__) && (__clang_major__ >= 12))
  bool HasP10Instructions = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
  if (HasP10Instructions) {
    this->GemmFloatKernel = MlasSgemmKernelPOWER10;
    this->GemmDoubleKernel = MlasDgemmKernelPOWER10;
    this->GemmU8X8Dispatch = &MlasGemm8X8DispatchPOWER10;
  }
#endif
#endif

#endif  // __linux__
#endif  // MLAS_TARGET_POWER
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
