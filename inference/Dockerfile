# Multi-stage build to minimize final image size
# Build stage - includes full CUDA toolkit for compilation
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 

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
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python


# Set up CUDA library paths for building
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs

# Create symlink for CUDA driver stub (needed for linking during build)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Build llama.cpp with compatible version
RUN cd / && git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    # Use a more recent version with better gpt-oss support
    git checkout tags/b6451 && \
    git checkout -b llmmll

WORKDIR /llama.cpp
RUN if ! grep -q "gpt-oss" ./src/llama-arch.cpp; then exit 1; fi

RUN cmake -B build \
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
    pip wheel --wheel-dir /opt/wheels torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 

RUN CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75;86 -DCMAKE_PREFIX_PATH=/llama.cpp/build/bin" \
    LLAMA_CPP_LIB_PATH=/llama.cpp/build/bin \
    pip wheel --wheel-dir /opt/wheels llama-cpp-python==0.3.16 -vvv

# Clean up llama-cpp-python build artifacts to save space
# RUN find /tmp -name "*llama*" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean up build-time LD_LIBRARY_PATH to avoid runtime stub library conflicts
ENV LD_LIBRARY_PATH="/llama.cpp/build/bin:/usr/local/cuda/lib64"


# Runtime stage - minimal CUDA runtime
# FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    RUNNER_DIR="/app/runner" \
    SERVER_DIR="/app/server" \
    EVALUATION_DIR="/app/evaluation" \
    EVALUATION_VENV="/opt/venv/evaluation" \
    SERVER_VENV="/opt/venv/server" \
    RUNNER_VENV="/opt/venv/runner" \
    # llama.cpp runtime library locations (ensure dependent libs load)
    LLAMA_CPP_LIB_PATH="/llama.cpp/build/bin" \
    LLAMA_CPP_LIB="/llama.cpp/build/bin/libllama.so" \
    LD_LIBRARY_PATH="/llama.cpp/build/bin:/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HOME=/root/.cache/huggingface \
    BNB_CUDA_VERSION=128 \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="7.5;8.6" \
    # CUDA runtime setup (remove stubs from path)
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    CUDA_LAUNCH_BLOCKING=0 \
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

# Install only runtime dependencies (no build tools!) and create python symlink
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    binutils \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy pre-built artifacts from builder stage
# COPY --from=builder /tmp/llama.cpp /llama.cpp
# COPY --from=builder /opt/wheels /opt/wheels

# Ensure llama.cpp shared libs are discoverable by dynamic linker (POSIX sh compatible)
RUN for f in /llama.cpp/build/bin/libggml*.so /llama.cpp/build/bin/libllama.so; do \
    if [ ! -f "$f" ]; then echo "Missing expected library $f" >&2; exit 1; fi; \
    ln -sf "$f" /usr/local/lib/; \
    done; ldconfig || true

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
    pip install --no-cache-dir --find-links /opt/wheels --no-index torch torchaudio torchvision llama-cpp-python && \
    deactivate; \
    done && \
    . ${RUNNER_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/runner && \
    deactivate && \
    . ${SERVER_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/server && \
    deactivate && \
    . ${EVALUATION_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/evaluation && \
    deactivate

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup cross-environment access
RUN python3 /app/setup_cross_env_access.py

# Make scripts executable and setup CUDA runtime
RUN chmod +x /app/startup.sh /app/run_with_env.sh /app/run.sh /app/setup_cuda_runtime.sh && \
    mkdir -p /tmp/cuda_cache

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