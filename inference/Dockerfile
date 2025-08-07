# Multi-stage build to minimize final image size
# Build stage - includes full CUDA toolkit for compilation
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    build-essential \
    cmake \
    g++ \
    ninja-build \
    libcurl4-openssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*


# Set up CUDA library paths for building
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

# Create symlink for CUDA driver stub (needed for linking during build)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Build llama.cpp
RUN cd /tmp && git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    git checkout ab14019 && \
    git checkout -b llmmll && \
    cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;86" \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Create shared pip cache and install packages that need compilation
RUN mkdir -p /opt/pip-cache /opt/wheels

# Install PyTorch and other packages that need CUDA compilation
RUN python3 -m venv /tmp/build_venv && \
    . /tmp/build_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir /opt/wheels torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75;86 -DCMAKE_PREFIX_PATH=/llama.cpp/build/bin" \
    LLAMA_CPP_LIB_PATH=/tmp/llama.cpp/build/bin \
    pip wheel --wheel-dir /opt/wheels llama-cpp-python


# Runtime stage - minimal CUDA runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    EVALUATION_VENV="/opt/venv/evaluation" \
    SERVER_VENV="/opt/venv/server" \
    RUNNER_VENV="/opt/venv/runner" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HOME=/root/.cache/huggingface \
    BNB_CUDA_VERSION=128 \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="7.5;8.6" \
    # Default values for configurable environment variables
    LOG_LEVEL="debug" \
    # Server configuration
    PORT=8000 \
    # Image generation configuration
    IMAGE_DIR="/root/images" \
    IMAGE_GENERATION_ENABLED="true" \
    MAX_IMAGE_SIZE="2048" \
    IMAGE_RETENTION_HOURS="24" \
    # GRPC configuration
    INFERENCE_SERVICES_HOST="0.0.0.0" \
    INFERENCE_SERVICES_PORT="50051" \
    GRPC_MAX_WORKERS=10 \
    GRPC_MAX_MESSAGE_SIZE=104857600 \
    GRPC_MAX_CONCURRENT_RPCS=100 \
    GRPC_ENABLE_REFLECTION=true \
    GRPC_SECURITY_ENABLED=false \
    GRPC_USE_TLS=false

# Install only runtime dependencies (no build tools!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy pre-built artifacts from builder stage
COPY --from=builder /tmp/llama.cpp/* /llama-cpp/
COPY --from=builder /opt/wheels /opt/wheels

# Set working directory and copy application code
WORKDIR /app
COPY . .

# Create virtual environments
RUN python3 -m venv ${RUNNER_VENV} && \
    python3 -m venv ${SERVER_VENV} && \
    python3 -m venv ${EVALUATION_VENV}

# Install dependencies in a single layer to reduce image layers
RUN for venv in ${RUNNER_VENV} ${SERVER_VENV} ${EVALUATION_VENV}; do \
    . $venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --find-links /opt/wheels --no-index torch torchaudio torchvision && \
    deactivate; \
    done && \
    . ${RUNNER_VENV}/bin/activate && \
    LLAMA_CPP_LIB_PATH=/llama.cpp/build/bin pip install --no-cache-dir --find-links /opt/wheels --no-index llama-cpp-python && \
    pip install --no-cache-dir -e /app/runner && \
    deactivate && \
    . ${SERVER_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/server && \
    pip install --no-cache-dir "uvicorn[standard]" watchfiles websockets && \
    deactivate && \
    . ${EVALUATION_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/evaluation && \
    deactivate

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup cross-environment access
RUN python3 /app/setup_cross_env_access.py

# Make scripts executable
RUN chmod +x /app/startup.sh /app/run_with_env.sh

# Final cleanup - remove build artifacts and caches
RUN rm -rf /opt/wheels \
    && find /opt/venv -name "*.pyc" -delete \
    && find /opt/venv -name "__pycache__" -delete \
    && rm -rf /root/.cache/pip \
    && echo "# Auto-source run_with_env.sh" >> /root/.bashrc \
    && echo "source /app/run_with_env.sh" >> /root/.bashrc \
    && echo 'echo -e "\033[1;32m=== Inference Environment Ready ===\033[0m"' >> /root/.bashrc \
    && echo 'echo -e "\033[1;33mAvailable environments: runner, server, evaluation\033[0m"' >> /root/.bashrc \
    && echo 'echo "Use: v <environment> <command>"' >> /root/.bashrc

# Expose ports  
EXPOSE 11434 8000 50051

# Default command
CMD ["/app/startup.sh"]