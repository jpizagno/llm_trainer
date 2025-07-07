FROM huggingface/transformers-tensorflow-gpu

# Install necessary build tools and CUDA development dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-dev \
    cuda-toolkit-12-6 \
    && rm -rf /var/lib/apt/lists/*

    
# Set environment variables for CUDA 12.6
ENV CUDA_VERSION=126
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python deps
RUN pip3 install --upgrade pip setuptools
RUN pip3 install torch peft
RUN pip3 install bitsandbytes==0.42.0

# Clone and build bitsandbytes
RUN git clone https://github.com/TimDettmers/bitsandbytes.git && \
    cd bitsandbytes && \
    python3 setup.py install && \
    cd .. && rm -rf bitsandbytes

# Optional: Verify CUDA setup
RUN python3 -m bitsandbytes || true

COPY ./ /app/
WORKDIR /app/



CMD ["sleep", "infinity"]