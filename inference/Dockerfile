# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:25.04-py3


# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
RUN apt-get install -y libglib2.0-0
RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt
# Set up Hugging Face for model downloads
RUN pip install -U "huggingface_hub[cli]"


# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH" \
#   VIRTUAL_ENV="/opt/venv" \
#   PYTHONUNBUFFERED=1 \
#   IMAGE_DIR="/root/images"


# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh -o install.sh && \
  chmod +x install.sh && \
  sh install.sh

RUN pip install ninja
RUN pip install flash-attn --no-build-isolation
# Explicitly install sentencepiece for tokenizer support
RUN pip install sentencepiece

# Install bitsandbytes with CUDA support and ensure compatibility
RUN pip uninstall -y bitsandbytes
RUN CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- | cut -d. -f1-2 | tr -d ".") && \
  echo "Detected CUDA version: $CUDA_VERSION" && \
  pip install bitsandbytes>=0.43.0 pika>=1.3.2 && \
  # Create symlinks for CUDA compatibility
  BNB_DIR=$(pip show bitsandbytes | grep Location | awk '{print $2}')/bitsandbytes && \
  if [ -f "$BNB_DIR/libbitsandbytes_cuda$CUDA_VERSION.so" ]; then \
  cp "$BNB_DIR/libbitsandbytes_cuda$CUDA_VERSION.so" "$BNB_DIR/libbitsandbytes_cuda.so"; \
  else \
  echo "Warning: Cannot find CUDA binary for version $CUDA_VERSION. Creating a fallback symlink."; \
  for f in "$BNB_DIR"/libbitsandbytes_cuda*.so; do \
  if [ -f "$f" ]; then \
  ln -sf "$f" "$BNB_DIR/libbitsandbytes_cuda.so"; \
  echo "Created symlink from $(basename $f) to libbitsandbytes_cuda.so"; \
  break; \
  fi; \
  done; \
  fi

# Set working directory for our application
WORKDIR /app

# Copy app code
COPY . .

# Install application dependencies
RUN pip install -r requirements.txt 

# # Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  HF_HOME=/root/.cache/huggingface \
  PYTHONPATH="/app:${PYTHONPATH:-}" \
  PYTHONUNBUFFERED=1 \
  # Force bitsandbytes to use CUDA 12.8 libraries instead of 12.9
  BNB_CUDA_VERSION=128

# Directory for image storage
RUN mkdir -p /root/images

# Install uvicorn with extra dependencies
RUN pip install "uvicorn[standard]" watchfiles websockets

# Expose ports
EXPOSE 11434 8000

# Default command
CMD ["sh", "-c", "ollama serve & echo 'Note: For live code editing, mount your code to /app' && if [ -f /app/app.py ]; then cd /app && uvicorn app:app --host 0.0.0.0 --port 8000 --reload; else echo 'WARNING: app.py not found in /app directory'; ls -la /app; exit 1; fi"]
