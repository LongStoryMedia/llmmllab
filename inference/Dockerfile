# Multi-stage build to minimize final image size
# Build stage - includes full CUDA toolkit for compilation
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 as builder

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
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set up CUDA library paths for building
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs

# Create symlink for CUDA driver stub (needed for linking during build)
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Create wheels directory
RUN mkdir -p /opt/wheels

# Build only packages that need CUDA compilation - avoid PyTorch wheels in builder
RUN python3 -m venv /tmp/build_venv && \
    . /tmp/build_venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    # Install llama-cpp-python dependencies first
    pip install --no-cache-dir diskcache>=5.6.1 typing-extensions>=4.5.0 numpy>=1.20.0 jinja2>=2.11.3 && \
    # Only build llama-cpp-python with CUDA (the main package that needs compilation)
    CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75;86" \
    pip wheel --wheel-dir /opt/wheels git+https://github.com/LongStoryMedia/llama-cpp-python.git@feature/b6482 --no-deps -vvv && \
    # Clean up immediately to reduce layer size
    deactivate && \
    rm -rf /tmp/build_venv /root/.cache /tmp/*

# Runtime stage - minimal CUDA runtime
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Set environment variables - using shared environment approach
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    RUNNER_DIR="/app/runner" \
    SERVER_DIR="/app/server" \
    EVALUATION_DIR="/app/evaluation" \
    COMPOSER_DIR="/app/composer" \
    SHARED_VENV="/opt/venv/shared" \
    # Set all service environments to use shared venv
    EVALUATION_VENV="/opt/venv/shared" \
    SERVER_VENV="/opt/venv/shared" \
    RUNNER_VENV="/opt/venv/shared" \
    COMPOSER_VENV="/opt/venv/shared" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HOME=/root/.cache/huggingface \
    BNB_CUDA_VERSION=128 \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="7.5;8.6" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    CUDA_LAUNCH_BLOCKING=0 \
    PORT=8000 \
    IMAGE_DIR="/root/images" \
    IMAGE_GENERATION_ENABLED="true" \
    MAX_IMAGE_SIZE="2048" \
    IMAGE_RETENTION_HOURS="24" \
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
    binutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy only the compiled llama-cpp-python wheel from builder
COPY --from=builder /opt/wheels /opt/wheels

# Set working directory
WORKDIR /app

# Create single shared virtual environment instead of 4 separate ones
RUN python3 -m venv ${SHARED_VENV}

# Install all dependencies in shared environment to avoid duplication (before copying source code)
RUN . ${SHARED_VENV}/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install PyTorch runtime directly (much smaller than pre-built wheels)
    pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    # Install llama-cpp-python dependencies first
    pip install --no-cache-dir diskcache>=5.6.1 typing-extensions>=4.5.0 numpy>=1.20.0 jinja2>=2.11.3 && \
    # Install our custom llama-cpp-python from wheels
    pip install --no-cache-dir --find-links /opt/wheels --no-index llama-cpp-python && \
    # Cleanup wheels and pip cache
    rm -rf /opt/wheels /root/.cache/pip && \
    deactivate

# Copy application code AFTER installing base dependencies
COPY . .

# Install service packages in development mode (these will rebuild when code changes)
RUN . ${SHARED_VENV}/bin/activate && \
    pip install --no-cache-dir -e /app/runner -e /app/server -e /app/evaluation -e /app/composer && \
    deactivate

# Make scripts executable and setup CUDA runtime
RUN chmod +x /app/startup.sh /app/run_with_env.sh /app/run.sh /app/setup_cuda_runtime.sh && \
    mkdir -p /tmp/cuda_cache

# Final cleanup - comprehensive cleanup to minimize image size
RUN find ${SHARED_VENV} -name "*.pyc" -delete && \
    find ${SHARED_VENV} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find ${SHARED_VENV} -name "*.pyo" -delete && \
    find ${SHARED_VENV} -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find ${SHARED_VENV} -name "test" -type d -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/* && \
    echo "# Auto-source run_with_env.sh" >> /root/.bashrc && \
    echo "source /app/run_with_env.sh" >> /root/.bashrc && \
    echo 'echo -e "\033[1;32m=== Inference Environment Ready (Shared) ===\033[0m"' >> /root/.bashrc && \
    echo 'echo -e "\033[1;33mUsing shared environment for all services\033[0m"' >> /root/.bashrc && \
    echo 'echo "Use: v <environment> <command>"' >> /root/.bashrc

# Expose ports  
EXPOSE 8000 50051

# Default command
CMD ["sh", "-c", "cd /app/server && /app/v.sh python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app --timeout-graceful-shutdown 0"]