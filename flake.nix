# luxtts.cpp — Nix flake: canonical build environment for the C++/ggml TTS engine.
#
# The engine is CPU-only; everything here is CPU-side toolchain.
#
#   nix develop            # dev shell: cmake + gcc + ninja + espeak-ng
#   nix build .            # hermetic CPU build -> build/<hash>/bin/luxtts
#                          #   (nixpkgs ggml + kissfft, no network, no git)
#
# In the shell (see also README):
#   cmake -B build -DCMAKE_BUILD_TYPE=Release   # CPU
#
# espeak-ng (the G2P dependency) is provided by the dev shell; CMake
# auto-detects the prefix, or pin it with -DLUX_ESPEAK_DIR=<prefix>.

{
  description = "luxtts.cpp — C++/ggml LuxTTS inference; dev shell + CPU build";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s} s);

      src = ./.;

      # Toolchain: build tools + espeak-ng >= 1.52 (the G2P terminator API the
      # tokenizer uses). No git: the build is hermetic — ggml + kissfft come
      # from the locked nixpkgs (LUX_SYSTEM_DEPS=ON), no FetchContent, no network.
      buildInputs = pkgs: with pkgs; [
        cmake
        ninja
        gnumake
        gcc
        pkgs.espeak-ng
      ];

      # Hermetic third-party deps, all from the locked nixpkgs:
      # ggml: static + OpenMP, with GGML_MAX_NAME=128 (the project's GGUF tensor
      # names need it; must match the project's add_compile_definitions(
      # GGML_MAX_NAME=128)). Single `out` output: include/ + lib/ + lib/cmake/ggml.
      ggmlPkg = pkgs: pkgs.ggml.overrideAttrs (final: prev: prev // {
        cmakeFlags = (prev.cmakeFlags or []) ++ [
          "-DBUILD_SHARED_LIBS=OFF"
          "-DGGML_OPENMP=ON"
          "-DCMAKE_C_FLAGS=-DGGML_MAX_NAME=128"
          "-DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=128"
        ];
      });

      # kissfft: float static lib (the nixpkgs default is shared; enableStatic
      # true). out: lib/libkissfft-float.a; dev: include/kissfft/ + CMake config.
      kissfftPkg = pkgs: pkgs.kissfftFloat.override { enableStatic = true; };

      # A build of the project. Hermetic: LUX_SYSTEM_DEPS=ON links the nixpkgs
      # ggml/kissfft outputs (no network at build time; the daemon enforces
      # sandbox=true, so this must stay hermetic).
      buildPackage = pkgs:
        let
          ggml = ggmlPkg pkgs;
          kiss = kissfftPkg pkgs;
        in
        pkgs.stdenv.mkDerivation {
          pname = "luxtts-cpp";
          version = "0.1";
          src = pkgs.lib.cleanSource src;
          nativeBuildInputs = buildInputs pkgs ++ [ ggml kiss.out kiss.dev ];
          # Note: dontConfigure alone. Do NOT add dontBuild — stdenv's
          # runPhase checks dontBuild BEFORE evaluating a custom buildPhase
          # variable, so dontBuild=true would silently skip our cmake build
          # and installPhase would fail on the missing binary.
          dontConfigure = true;
          # CMake flags follow the CMakeLists.txt defaults (LUX_NATIVE=ON,
          # LUX_BUILD_CLI=ON, LUX_SHARED=OFF); CMAKE_BUILD_TYPE defaults to
          # Release there too, so it is the only one passed explicitly.
          # Hermetic deps: LUX_SYSTEM_DEPS=ON + the nix prefixes
          #   -DCMAKE_PREFIX_PATH -> nix ggml (find_package(ggml))
          #   -DLUX_KISSFFT_ROOT  -> nix kissfft-float static outputs
          #   -DLUX_ESPEAK_DIR    -> nix espeak-ng (lib + share/espeak-ng-data)
          # LUX_ESPEAK_DATA_DIR stays at the bundled third_party/espeak-ng-data
          # (tracked in the source tree, so cleanSource keeps it).
          buildPhase = ''
            cmake -S . -B $TMPDIR/build -DCMAKE_BUILD_TYPE=Release \
              -DLUX_SYSTEM_DEPS=ON \
              -DCMAKE_PREFIX_PATH=${ggml} \
              -DLUX_KISSFFT_ROOT='${kiss.out};${kiss.dev}' \
              -DLUX_ESPEAK_DIR=${pkgs.espeak-ng}
            cmake --build $TMPDIR/build -j $NIX_BUILD_CORES
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp $TMPDIR/build/cli/luxtts $out/bin/
          '';
        };
    in
    {
      devShells = forAll (pkgs: _system: {
        default = pkgs.mkShell {
          packages = buildInputs pkgs;
          # pin espeak-ng to this shell's 1.52 (CMake auto-detect may find an
          # older store copy lacking the G2P terminator API)
          env.LUX_ESPEAK_DIR = toString pkgs.espeak-ng;
          shellHook = ''
            echo "luxtts.cpp dev shell"
            echo ""
            echo "  hermetic CPU build (no network): nix build ."
            echo "  host build (FetchContent): cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
            echo ""
            echo "  LUX_ESPEAK_DIR is pre-set for this shell."
          '';
        };
      });

      packages = forAll (pkgs: _system: {
        luxtts-cpp = buildPackage pkgs;
        default = buildPackage pkgs;
      });
    };
}
