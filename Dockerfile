# ───────────────────────────────────────────────────────────── #
# Dockerfile for running CELLFLOW (TAP) pipeline in a container #
# ───────────────────────────────────────────────────────────── #

FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV CROP_SIZE=64x64
ENV PIXEL_RES=0.65

# Install system packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    openjdk-17-jre-headless \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy pandas matplotlib seaborn scikit-image imageio tqdm pyyaml \
        gitpython configargparse tifffile opencv-python typing-extensions \
        torch torchvision torchaudio

# Install Nextflow
RUN curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod +x /usr/local/bin/nextflow

# Set working directory and copy all necessary files
WORKDIR /app
COPY . /app

# Ensure these folders are explicitly copied if .dockerignore excludes them
COPY Workflow /app/Workflow
COPY TAP /app/TAP

# Make helper scripts executable
RUN chmod +x /app/run_tap.sh

# Default shell
CMD ["bash"]
