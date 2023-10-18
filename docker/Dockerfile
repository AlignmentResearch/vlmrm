FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 as base

ARG NVIDIA_DRIVER_VERSION=525
RUN apt update && apt install -y --no-install-recommends \
    libxau6 \
    libxdmcp6 \
    libxcb1 \
    libxext6 \
    libx11-6 && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility,video
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=device

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt update && apt install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libnvidia-gl-${NVIDIA_DRIVER_VERSION} && \
    rm -rf /var/lib/apt/lists/*

FROM base

WORKDIR /root/vlmrm
ENV EGL_DEVICE_ID=1

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes \
    git \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    python3-dev \
    # for box2d
    swig \
    # for pygame
    xvfb \
    # for mujoco
    mesa-utils
# Link files from current dir
COPY . /root/vlmrm
# Install vlmrm
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -e /root/vlmrm
RUN git config --global --add safe.directory /root/vlmrm
