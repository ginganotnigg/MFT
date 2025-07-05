FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch and other dependencies
RUN pip install -r requirements.txt

# Copy your code and model files
COPY . .

# Set Hugging Face cache and offline mode
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=1

# Expose FastAPI port
EXPOSE 7777

# Default command: run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7777"]