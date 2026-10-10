# luxtts.cpp — self-contained C++/ggml port of the LuxTTS CPU inference pipeline.
#
# CPU-only. No GPU backends are exposed (CUDA/Vulkan/Metal/HIP/SYCL are forced OFF
# for the ggml dependency). Weights are loaded from a GGUF file produced by
# tools/convert_to_gguf.py; there is no onnxruntime dependency anywhere.
#
# Reference architecture: ggml graph per module, weights in GGUF, C++ static lib + CLI + flat C API.

cmake_minimum_required(VERSION 3.20)
project(luxtts-cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo)
endif()

# ---------------------------------------------------------------------------
# Options (CPU-only — do NOT add GPU toggles here; the port is CPU-only by design)
# ---------------------------------------------------------------------------
option(LUX_SHARED      "Build libluxtts as a shared library" OFF)
option(LUX_NATIVE      "Compile with -march=native"         ON)
option(LUX_BUILD_CLI   "Build the luxtts CLI"                ON)

# ---------------------------------------------------------------------------
# Third-party — either fetched at configure time (see third_party/README.md)
# or host-provided: -DLUX_SYSTEM_DEPS=ON (flake.nix) links nixpkgs ggml +
# kissfft outputs instead of FetchContent (hermetic, no network at build time).
# ---------------------------------------------------------------------------
option(LUX_SYSTEM_DEPS
       "Link host-provided ggml + kissfft (find_package / imported target) instead of FetchContent (used by flake.nix)" OFF)

# The GGUF keeps torch state-dict tensor names verbatim; the longest
# (~73 chars) exceeds ggml's default GGML_MAX_NAME=64. It sizes
# ggml_tensor::name, so it must be identical in ggml and in every consumer TU —
# hence a directory-scope definition ahead of the ggml FetchContent. In
# LUX_SYSTEM_DEPS mode flake.nix builds the nix ggml with the same
# -DGGML_MAX_NAME=128 (see flake.nix's ggml overrideAttrs).
add_compile_definitions(GGML_MAX_NAME=128)

# Keep ggml (and everything else) static so the libluxtts static target embeds
# the engine. A plain CACHE default (no FORCE) still honours an explicit
# -DBUILD_SHARED_LIBS=ON on the command line.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries (ggml is linked statically into luxtts by default)")

if(LUX_SYSTEM_DEPS)
  # --- Host-provided deps (nixpkgs outputs, wired up by flake.nix) --------
  # ggml: the nix prefix is on CMAKE_PREFIX_PATH; its ggml-config.cmake
  # (lib/cmake/ggml/) provides the namespaced imported targets, incl.
  # ggml::ggml-base and ggml::ggml-cpu (the CPU-only backend set the nix
  # package builds).
  find_package(ggml REQUIRED)
  set(LUX_GGML_TARGETS ggml::ggml-base ggml::ggml-cpu)

  # kissfft: the existing link lines expect a target named `kissfft`, so create
  # an IMPORTED STATIC one from the nixpkgs kissfft-float (static) outputs:
  #   out: lib/libkissfft-float.a
  #   dev: include/kissfft/*.h  (headers are namespaced under kissfft/)
  # LUX_KISSFFT_ROOT is a ';'-separated list of those outputs (the flake
  # passes both); the lib and the headers are located inside it.
  set(LUX_KISSFFT_ROOT "" CACHE STRING
      "kissfft-float (static) nix output(s), ';'-separated; required with LUX_SYSTEM_DEPS=ON")
  if(NOT LUX_KISSFFT_ROOT)
    message(FATAL_ERROR "LUX_SYSTEM_DEPS=ON requires -DLUX_KISSFFT_ROOT=<out>;<dev> (nixpkgs kissfft-float static outputs)")
  endif()
  set(_lux_kiss_lib_paths)
  set(_lux_kiss_inc_dirs)
  foreach(_lux_r IN LISTS LUX_KISSFFT_ROOT)
    list(APPEND _lux_kiss_lib_paths "${_lux_r}" "${_lux_r}/lib")
    if(IS_DIRECTORY "${_lux_r}/include/kissfft")
      list(APPEND _lux_kiss_inc_dirs "${_lux_r}/include/kissfft")
    endif()
  endforeach()
  find_library(LUX_KISSFFT_LIB NAMES kissfft-float
    HINTS ${_lux_kiss_lib_paths} NO_DEFAULT_PATH)
  if(NOT LUX_KISSFFT_LIB)
    message(FATAL_ERROR "kissfft-float static lib not found in LUX_KISSFFT_ROOT=${LUX_KISSFFT_ROOT}")
  endif()
  if(NOT _lux_kiss_inc_dirs)
    message(FATAL_ERROR "kissfft headers (include/kissfft/) not found in LUX_KISSFFT_ROOT=${LUX_KISSFFT_ROOT}")
  endif()
  add_library(kissfft STATIC IMPORTED GLOBAL)
  set_target_properties(kissfft PROPERTIES
    IMPORTED_LOCATION "${LUX_KISSFFT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${_lux_kiss_inc_dirs}"
    # nixpkgs builds the float variant: scalar type must match the lib.
    INTERFACE_COMPILE_DEFINITIONS "kiss_fft_scalar=float"
    INTERFACE_LINK_LIBRARIES "m")
else()
  # --- Host build: fetch pinned ggml + kissfft at configure time -----------
  include(FetchContent)

  # ggml: the CPU tensor engine. CPU-only build; every accelerator backend is OFF.
  set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE         ${LUX_NATIVE} CACHE BOOL "" FORCE)
  set(GGML_OPENMP         ON  CACHE BOOL "" FORCE)   # CPU threading for the graph compute
  set(GGML_BLAS           OFF CACHE BOOL "" FORCE)   # no external BLAS; pure CPU
  # GPU / accelerator backends — all OFF (hard constraint: CPU-only).
  set(GGML_CUDA    OFF CACHE BOOL "" FORCE)
  set(GGML_VULKAN  OFF CACHE BOOL "" FORCE)
  set(GGML_METAL   OFF CACHE BOOL "" FORCE)
  set(GGML_HIP     OFF CACHE BOOL "" FORCE)
  set(GGML_SYCL    OFF CACHE BOOL "" FORCE)
  set(GGML_OPENCL  OFF CACHE BOOL "" FORCE)
  set(GGML_WEBGPU  OFF CACHE BOOL "" FORCE)
  set(GGML_VIRTGPU OFF CACHE BOOL "" FORCE)
  set(GGML_ZDNN    OFF CACHE BOOL "" FORCE)
  set(GGML_ZENDNN  OFF CACHE BOOL "" FORCE)
  set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
  set(GGML_ET      OFF CACHE BOOL "" FORCE)
  set(GGML_OPENVINO OFF CACHE BOOL "" FORCE)
  # PIN: pin to a known-good ggml commit for reproducible builds.
  # v0.24.0 — commit 456172ec (2026-09; "ggml: bump version to 0.24.0").
  FetchContent_Declare(
    ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml
    GIT_TAG        456172ec733a135778adcd32d00e576a58232e45
  )
  FetchContent_MakeAvailable(ggml)

  # kissfft: 1024-pt real FFT for the Linkwitz-Riley crossover in the vocoder.
  # Pinned to a known-good commit (kissfft 131.2.0, commit 7bce4153; the tag is
  # `131.2.0` without a v-prefix). Static float build; tests/tools/pkg-config disabled.
  set(KISSFFT_STATIC   ON  CACHE BOOL "" FORCE)
  set(KISSFFT_TEST     OFF CACHE BOOL "" FORCE)
  set(KISSFFT_TOOLS    OFF CACHE BOOL "" FORCE)
  set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft
    GIT_TAG        7bce4153c6bc8aba2db0e889e576f9d00505cbe1
  )
  FetchContent_MakeAvailable(kissfft)

  set(LUX_GGML_TARGETS ggml-base ggml-cpu)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(LUX_SOURCES
  src/common.cpp
  src/tokenizer.cpp
  src/zipformer.cpp
  src/sampling.cpp
  src/vocos.cpp
  src/audio_io.cpp
  src/fbank.cpp
  src/luxtts.cpp
)

if(LUX_SHARED)
  add_library(luxtts SHARED ${LUX_SOURCES})
  set_target_properties(luxtts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(luxtts STATIC ${LUX_SOURCES})
endif()

target_include_directories(luxtts PUBLIC include)
target_include_directories(luxtts PRIVATE src third_party)

# ggml CPU engine + kissfft (real FFT). Link the base + cpu targets (magpie
# pattern); the target set depends on the dep mode (in-tree vs nix imported).
target_link_libraries(luxtts PUBLIC ${LUX_GGML_TARGETS})
target_link_libraries(luxtts PRIVATE kissfft)

target_compile_options(luxtts PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)

find_package(Threads REQUIRED)
target_link_libraries(luxtts PUBLIC Threads::Threads)

# OpenMP: the host-side vocos loops (conv1d / norms / ISTFT / resample) and
# the zipformer host round-trips parallelize with omp pragmas; thread count
# follows OMP_NUM_THREADS (the ggml graphs get an explicit n_threads arg).
find_package(OpenMP REQUIRED)
target_link_libraries(luxtts PUBLIC OpenMP::OpenMP_CXX)

# ---------------------------------------------------------------------------
# espeak-ng — English G2P (required, CPU-only). espeak-ng is an unconditional
# dependency (the reference piper_phonemize treats it as one), so
# g2p_espeak_en_us() always compiles against the real library. The prefix
# resolves via LUX_ESPEAK_DIR (pin) or auto-detection under /nix/store; the
# data dir is pinned via LUX_ESPEAK_DATA_DIR (the runtime dictionary dir,
# share/espeak-ng-data).
# ---------------------------------------------------------------------------
set(LUX_ESPEAK_DIR "" CACHE PATH
    "espeak-ng prefix with include/, lib/, share/espeak-ng-data/ (empty = auto-detect nix store)")

if(LUX_ESPEAK_DIR)
  set(LUX_ESPEAK_RESOLVED ${LUX_ESPEAK_DIR})
  if(NOT EXISTS "${LUX_ESPEAK_RESOLVED}/lib/libespeak-ng.so"
     OR NOT EXISTS "${LUX_ESPEAK_RESOLVED}/share/espeak-ng-data/en_dict")
    message(FATAL_ERROR "LUX_ESPEAK_DIR=${LUX_ESPEAK_DIR} lacks lib/libespeak-ng.so and/or share/espeak-ng-data/en_dict")
  endif()
else()
  set(LUX_ESPEAK_RESOLVED "")
  file(GLOB _lux_espeak_dirs "/nix/store/*-espeak-ng-*")
  foreach(_d IN LISTS _lux_espeak_dirs)
    if(IS_DIRECTORY "${_d}"
       AND EXISTS "${_d}/lib/libespeak-ng.so"
       AND EXISTS "${_d}/share/espeak-ng-data/en_dict")
      if(LUX_ESPEAK_RESOLVED STREQUAL "")
        set(LUX_ESPEAK_RESOLVED "${_d}")
      else()
        message(WARNING "luxtts: multiple nix-store espeak-ng found; using ${LUX_ESPEAK_RESOLVED} (set LUX_ESPEAK_DIR to pin)")
      endif()
    endif()
  endforeach()
  if(LUX_ESPEAK_RESOLVED STREQUAL "")
    message(FATAL_ERROR "espeak-ng not found (set LUX_ESPEAK_DIR to an espeak-ng prefix)")
  endif()
endif()

# The reference (piper_phonemize) is trained against a specific espeak-ng-data
# (dictionary + phondata). A nix-store espeak prefix ships its OWN data, which
# can emit different IPA for the same word (e.g. "more" -> ɔː vs oː), silently
# breaking the token stream. LUX_ESPEAK_DATA_DIR pins the data dir so the G2P
# matches the reference; empty = use the prefix's bundled share/espeak-ng-data.
set(LUX_ESPEAK_DATA_DIR "" CACHE PATH
    "espeak-ng-data dir to use for G2P (empty = bundled third_party/espeak-ng-data, else the prefix's)")
if(NOT LUX_ESPEAK_DATA_DIR)
  # Default to the bundled reference data (matches piper_phonemize, the
  # reference G2P); a nix-store espeak prefix's own data can emit different
  # IPA for the same word and silently break the token stream.
  set(LUX_ESPEAK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/espeak-ng-data")
endif()
if(NOT EXISTS "${LUX_ESPEAK_DATA_DIR}/en_dict")
  message(FATAL_ERROR "LUX_ESPEAK_DATA_DIR=${LUX_ESPEAK_DATA_DIR} lacks en_dict")
endif()
# PUBLIC: static libluxtts consumers (CLI, tests) must see the espeak lib
# and the compile definitions at their link/compile time.
target_include_directories(luxtts PUBLIC "${LUX_ESPEAK_RESOLVED}/include")
target_link_libraries(luxtts PUBLIC "${LUX_ESPEAK_RESOLVED}/lib/libespeak-ng.so")
target_compile_definitions(luxtts PUBLIC LUX_ESPEAK_DATA_DIR="${LUX_ESPEAK_DATA_DIR}")
# rpath so built executables find libespeak-ng.so at run time (nix store):
set_target_properties(luxtts PROPERTIES
  BUILD_RPATH "${LUX_ESPEAK_RESOLVED}/lib"
  INSTALL_RPATH "${LUX_ESPEAK_RESOLVED}/lib")
message(STATUS "luxtts: espeak-ng G2P enabled: ${LUX_ESPEAK_RESOLVED}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(LUX_BUILD_CLI)
  add_executable(luxtts_cli cli/luxtts.cpp)
  target_link_libraries(luxtts_cli PRIVATE luxtts)
  set_target_properties(luxtts_cli PROPERTIES
    OUTPUT_NAME luxtts
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cli)
endif()
