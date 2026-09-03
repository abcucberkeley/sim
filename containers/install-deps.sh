#!/usr/bin/env bash
# Provision the SIRIUS build/runtime toolchain on top of an nvidia/cuda
# "devel" image. Shared by the Dockerfile (local development) and the
# Apptainer definition (cluster) so both environments are provisioned by the
# same script and stay in lockstep.
#
# Everything SIRIUS itself needs beyond this (Eigen, libtiff, zlib, FFTW,
# toml++, Catch2, nanobind, nvTIFF, nvCOMP) is fetched and built by CMake at
# configure time (cmake/Dependencies.cmake, cmake/NvidiaRedist.cmake), so the
# image only carries compilers, CMake, MPI and Python.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    libopenmpi-dev \
    openmpi-bin \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    xz-utils
rm -rf /var/lib/apt/lists/*

# Python deps used by the tests and benchmarks (numpy is the only hard one).
python3 -m pip install --no-cache-dir --break-system-packages \
    numpy \
    tifffile \
    "scikit-build-core>=0.10" \
    "cmake>=3.25" \
    ninja

# Sanity: CUDA toolkit present (the base image provides it) and cmake >= 3.25.
nvcc --version | tail -1
cmake --version | head -1
