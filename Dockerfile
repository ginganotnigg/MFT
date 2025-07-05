FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

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