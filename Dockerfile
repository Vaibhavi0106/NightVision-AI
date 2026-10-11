# Dockerfile for NightVision AI (GPU CUDA Accelerated Deployment)
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set non-interactive environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install System Dependencies & Python 3.10 + Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Copy Requirements and Install Backend Python Dependencies
COPY backend/requirements.txt ./backend/
RUN pip3 install --no-cache-dir -r backend/requirements.txt

# Copy Frontend Dependencies and Build Dist
COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm install
COPY frontend/ ./
RUN npm run build

# Copy Entire Repository
WORKDIR /app
COPY . .

# Expose Port 8000
EXPOSE 8000

# Run Unified Production Server
CMD ["python3", "start_production.py"]
