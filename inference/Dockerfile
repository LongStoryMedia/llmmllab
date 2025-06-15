# Stage 1: Build dependencies (minimal base image)
FROM python:3.9-slim AS builder

# Install build tools and curl
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl

# Create a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install pip in the virtual environment
RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

# Copy requirements and install dependencies in the builder stage
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Build Ollama
FROM ubuntu:22.04 AS ollama-builder
RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh -o install.sh
RUN chmod +x install.sh
RUN sh install.sh

# Stage 3: Final image with Ollama and app code
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Install Python and other required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-pip \
  python3-venv \
  python3-distutils \
  curl \
  git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app code first (to use requirements.txt)
COPY . .

# Create a virtual environment and install dependencies directly in the final stage
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
  VIRTUAL_ENV="/opt/venv" \
  PYTHONUNBUFFERED=1

# Install packages in the final stage
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "huggingface_hub[cli]"

# Copy Ollama from the ollama-builder stage
COPY --from=ollama-builder /usr/local/bin/ollama /usr/local/bin/
COPY --from=ollama-builder /usr/local/lib/ollama /usr/local/lib/ollama

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
  NVIDIA_VISIBLE_DEVICES=all \
  NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  HF_HOME=/root/.cache/huggingface \
  PYTHONPATH="/app"

RUN mkdir -p /app/images

# Verify our Python environment works and has all required packages
RUN python3 -m pip list | grep -e fastapi -e uvicorn

# Expose ports
EXPOSE 11434 8000

# Start services (run in foreground)
CMD ["sh", "-c", "ollama serve & uvicorn app:app --host 0.0.0.0 --port 8000"]
