# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM tensorflow/tensorflow:latest-gpu

ARG uid
ARG user

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    htop \
    tmux \
    rsync \
    zip \
    unzip \
    patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/
    
RUN mkdir -p ~/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C ~/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}


RUN pip3 install --upgrade pip \
	&& pip3 install --ignore-installed PyYAML
RUN pip3 install wandb \
    rlds \
    tensorflow==2.11.0 \
	  tfds-nightly==4.8.3.dev202303130045 \
    dm-acme==0.2.4 \
    dm-sonnet==2.0.1 \
    dm-reverb==0.10.0 \
	"jax[cuda11_cudnn82]==0.4.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    flax==0.6.4 \
    chex==0.1.6 \
    optax==0.1.4 \
    distrax==0.1.3 \
	free-mujoco-py==2.1.6 \
	ml_collections 

RUN pip3 install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

RUN git clone https://github.com/Farama-Foundation/d4rl /d4rl
RUN git clone https://github.com/aravindr93/mjrl /mjrl
WORKDIR /mjrl 
RUN pip3 install . 
WORKDIR /d4rl 
RUN sed "71d" setup.py > tmp.py \
	&& rm setup.py \
    && mv tmp.py setup.py \
    && pip3 install .

ENV PYTHONPATH /offbench:${PYTHONPATH}

WORKDIR /offbench
