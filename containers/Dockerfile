# SIRIUS development image: CUDA 13.1 toolchain + the build dependencies.
#
#   docker build -f containers/Dockerfile -t sirius:dev .
#   docker run --rm -it --gpus all -v "$PWD":/work -w /work sirius:dev \
#       bash -c "cmake --preset linux-cuda-dev && cmake --build --preset linux-cuda-dev && ctest --preset linux-cuda-dev"
#
# The same base image and provisioning script back containers/sirius.def, so
# an Apptainer image built for the cluster is the same environment. You can
# also convert this image directly: apptainer build sirius.sif docker-daemon://sirius:dev
#
# Build args let the cluster image pin a different CUDA line (e.g. 12.8 for a
# driver that predates CUDA 13) without editing the file.
ARG CUDA_VERSION=13.1.0
ARG UBUNTU_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

COPY containers/install-deps.sh /tmp/install-deps.sh
RUN bash /tmp/install-deps.sh && rm /tmp/install-deps.sh

# CMake finds this before any distro nvcc (see cmake/ProjectOptions.cmake).
ENV CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /work
CMD ["bash"]
