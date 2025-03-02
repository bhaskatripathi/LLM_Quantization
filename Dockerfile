FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for HF cache
ENV HF_HOME=/tmp/huggingface_cache

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp /app/llama.cpp

# Build llama.cpp
WORKDIR /app/llama.cpp
RUN mkdir -p build && cd build && cmake .. && make

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Ensure the cache directory exists
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME

# Set the entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]