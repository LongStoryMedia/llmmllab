# Use NVIDIA CUDA development image as base image
# FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
# FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
  PYTHONPATH="/app" \
  IMAGE_DIR="/root/images" \
  EVALUATION_VENV="/opt/venv/evaluation" \
  SERVER_VENV="/opt/venv/server" \
  RUNNER_VENV="/opt/venv/runner" \
  BASE_VENV="/opt/venv/base" \
  DEBIAN_FRONTEND=noninteractive \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  HF_HOME=/root/.cache/huggingface \
  PORT=8000 \
  GRPC_PORT=50051 \
  GRPC_MAX_WORKERS=10 \
  GRPC_MAX_MESSAGE_SIZE=104857600 \
  GRPC_MAX_CONCURRENT_RPCS=100 \
  GRPC_ENABLE_REFLECTION=true \
  GRPC_SECURITY_ENABLED=false \
  GRPC_USE_TLS=false \
  BNB_CUDA_VERSION=129 \
  TOKENIZERS_PARALLELISM=true \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  python3-venv \
  curl \
  libcurl4-openssl-dev \
  git \
  wget \
  build-essential \
  cmake \
  # libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  g++ \
  ninja-build \
  ffmpeg \
  tree \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-9 nvidia-open

# Install basic tools
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /usr/local/cuda-12.8/compat/libcuda.so.1 /usr/local/cuda-12.8/compat/libcuda.so 
# RUN apt-get update && apt-get install -y git tree ffmpeg wget && \
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so 
#   apt-get install -y libglib2.0-0 python3-venv && \
# RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt
# RUN ls /lib64
# RUN ls /usr/local/
# RUN ls /usr/local/cuda-12.8/compat/
# Temporarily remove the version constraint for this terminal session
# RUN pip install --no-cache-dir --upgrade pip setuptools wheel pip-tools && \
#   pip install --no-cache-dir --upgrade "h11==0.16.0" && \
#   pip install --no-cache-dir -r /app/torch_requirements.txt 


# Set working directory for our application
WORKDIR /app

# Copy project files and install requirements for each environment
COPY . .
# Make startup script executable
RUN chmod +x /app/startup.sh

# Set up Hugging Face for model downloads
# RUN pip install -U "huggingface_hub[cli]"

# Build and install llama.cpp
WORKDIR /
# ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat/libcuda.so.1



# Temporarily remove the version constraint for this terminal session
# RUN unset PIP_CONSTRAINT

# # Install the desired version of fsspec
# # The quotes are necessary to prevent the shell from interpreting '<'
# RUN pip install "fsspec<=2023.12.0"

RUN git clone https://github.com/ggerganov/llama.cpp.git 
WORKDIR /llama.cpp 
# use commit ab14019 - per https://github.com/ggml-org/llama.cpp/issues/14847
# RUN git checkout ab14019 
# RUN git checkout -b llmmll 
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86" 
RUN cmake --build build --config Release -j$(nproc)

# Create virtual environments for each project
RUN mkdir -p /opt/venv && \
  python3 -m venv ${EVALUATION_VENV} && \
  python3 -m venv ${SERVER_VENV} && \
  python3 -m venv ${RUNNER_VENV} && \
  python3 -m venv ${BASE_VENV} && \
  # Install basic packages in the base environment
  . ${BASE_VENV}/bin/activate && \
  pip install --no-cache-dir --upgrade pip setuptools wheel pip-tools && \
  pip install --no-cache-dir torch torchaudio torchvision && \
  deactivate

# Set environment variables for llama-cpp-python
ENV LLAMA_CPP_LIB_PATH=/llama.cpp/build/bin \
  FORCE_CMAKE=1 \
  CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=75;86 -DCMAKE_PREFIX_PATH=/llama.cpp/build/bin"

# Install llama-cpp-python in the runner environment
RUN . ${RUNNER_VENV}/bin/activate && \
  CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75;86 -DCMAKE_PREFIX_PATH=/llama.cpp/build/bin" \
  pip install llama-cpp-python --no-cache-dir --force-reinstall  --no-binary llama-cpp-python && \
  deactivate


# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh -o install.sh && \
  chmod +x install.sh && \
  sh install.sh && \
  rm install.sh

# Install CUDA-specific packages in the base environment
RUN . ${BASE_VENV}/bin/activate && \
  pip install --no-cache-dir ninja packaging && \
  TORCH_CUDA_ARCH_LIST="7.5;8.6" pip install --no-cache-dir flash-attn && \
  pip install --no-cache-dir sentencepiece && \
  deactivate


# Install requirements for runner project
RUN . ${RUNNER_VENV}/bin/activate && \
  pip install --no-cache-dir --upgrade pip && \
  pip install setuptools wheel pip-tools build && \
  cd /app/runner && pip install -e . && \
  deactivate

# Install requirements for evaluation project
RUN . ${EVALUATION_VENV}/bin/activate && \
  pip install --no-cache-dir --upgrade pip && \
  pip install setuptools wheel pip-tools build && \
  cd /app/evaluation && pip install -e . && \
  deactivate

# Install requirements for server project
RUN . ${SERVER_VENV}/bin/activate && \
  pip install --no-cache-dir --upgrade pip && \
  pip install setuptools wheel pip-tools build && \
  cd /app/server && pip install -e . && \
  pip install "uvicorn[standard]" watchfiles websockets && \
  deactivate

WORKDIR /app

# Expose ports
# - 11434: Ollama
# - 8000: REST API server
# - 50051: gRPC server
EXPOSE 11434 8000 50051

# Add convenience script for running with specific environment
COPY run_with_env.sh /app/
RUN chmod +x /app/run_with_env.sh

# Default command to start all services
CMD ["/app/startup.sh"]
