# Use a CUDA-enabled base image
FROM nvidia/cuda:12.0.1-base-ubuntu20.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for GPU access
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Install dependencies without interactive prompts
RUN apt-get update && apt-get install -y \
  curl \
  git \
  python3-pip \
  python3-venv \
  python3.9 \
  python3.9-venv
# Set timezone (e.g., America/Chicago)
RUN ln -sf /usr/share/zoneinfo/America/Chicago /etc/localtime
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install Ollama (from the install script or binary)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up Python environment for Stable Diffusion
WORKDIR /app

# Copy the entire project at once
COPY . /app/

# Create Python virtual environment and install dependencies
RUN python3 -m venv venv
RUN . venv/bin/activate
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install -U peft
RUN pip install -r /app/requirements.txt

# Set the HuggingFace cache location
ENV HF_HOME=/root/.cache/huggingface

# Create cache directory if it doesn't exist
RUN mkdir -p ${HF_HOME}

# Create the images directory
RUN mkdir -p /app/images

# Expose ports
EXPOSE 11434 8000

# Start both services
CMD ["sh", "-c", "ollama serve & cd /app && . venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000"]
