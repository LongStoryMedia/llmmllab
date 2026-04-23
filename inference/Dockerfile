# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1 — compile llama.cpp with CUDA support
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    git wget curl \
    build-essential cmake g++ ninja-build \
    libcurl4-openssl-dev pkg-config ccache zstd \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ARG LLAMA_CPP_TAG=b8863
RUN git clone https://github.com/ggml-org/llama.cpp.git /llama.cpp \
    && cd /llama.cpp \
    && git checkout tags/${LLAMA_CPP_TAG} -b llmmll

WORKDIR /llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j6

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Pull the uv binary from the official image (pinned)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    SHARED_VENV="/opt/venv/shared" \
    UV_PROJECT_ENVIRONMENT="/opt/venv/shared" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/cuda/lib64" \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    HF_HOME=/root/.cache/huggingface \
    BNB_CUDA_VERSION=128 \
    TOKENIZERS_PARALLELISM=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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

# Runtime-only apt deps (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    curl libglib2.0-0 libgomp1 ffmpeg binutils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Compiled llama.cpp binaries
COPY --from=llama-builder /llama.cpp /llama.cpp

WORKDIR /app

# Create the shared venv uv will manage
RUN uv venv --python 3.12 ${SHARED_VENV}

# Install torch from the CUDA 12.1 index (special index, separate from uv sync).
# Kept as its own layer so python changes don't re-download torch wheels.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${SHARED_VENV}/bin/python \
    torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Copy only the dep manifests first so editing source doesn't bust the deps layer
COPY pyproject.toml uv.lock .python-version ./

# Resolve and install the rest of the deps into the shared venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Download Playwright Chromium browser binary + its system-level deps
RUN ${SHARED_VENV}/bin/playwright install --with-deps chromium

# Now copy application source
COPY . .

# Make scripts executable; prepare cuda cache dir
RUN chmod +x /app/startup.sh /app/run_with_env.sh /app/run.sh /app/setup_cuda_runtime.sh \
    && mkdir -p /tmp/cuda_cache

# Trim venv weight: drop bytecode/test trees that pip/uv install ships with packages
RUN find ${SHARED_VENV} -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find ${SHARED_VENV} -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} + \
    && find ${SHARED_VENV} \( -name "*.pyc" -o -name "*.pyo" \) -delete \
    && rm -rf /tmp/* /var/tmp/* \
    && echo "source /app/run_with_env.sh" >> /root/.bashrc

EXPOSE 8000

CMD ["/app/startup.sh"]
