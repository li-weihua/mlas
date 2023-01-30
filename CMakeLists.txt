cmake_minimum_required(VERSION 3.10)

project(MLAS ASM C CXX)

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLAS_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)

set(mlas_common_srcs
  ${MLAS_SRC_DIR}/platform.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/threading.cpp
)

# only build for linux
set(mlas_platform_srcs_sse2
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
  ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
  ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
  ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
)
set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

set(mlas_platform_srcs_avx
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
  ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
)
set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

set(mlas_platform_srcs_avx2
  ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
)
set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

set(mlas_platform_srcs
  ${mlas_platform_srcs_sse2}
  ${mlas_platform_srcs_avx}
  ${mlas_platform_srcs_avx2}
)

#add_library(mlas STATIC ${mlas_common_srcs} ${mlas_platform_srcs})
add_library(mlas SHARED ${mlas_common_srcs} ${mlas_platform_srcs})
target_include_directories(mlas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_compile_definitions(mlas PUBLIC BUILD_MLAS_NO_ONNXRUNTIME)

# test
add_executable(run test/test_sgemm.cc)
target_link_libraries(run PRIVATE mlas)
