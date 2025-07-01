FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Create working directory
WORKDIR /workspace

# Copy requirements (create this file in your repo)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your code and model files
COPY . .

# Set environment variables for Hugging Face cache (optional)
ENV HF_HOME=/workspace/.cache/huggingface

# Expose FastAPI port
EXPOSE 7777

# Default command: run FastAPI app (change app:app if your app is elsewhere)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7777"]