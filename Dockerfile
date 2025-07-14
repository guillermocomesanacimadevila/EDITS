# ───────────────────────────────────────────────────────────── #
# Dockerfile for running CELLFLOW (TAP) pipeline in a container #
# ───────────────────────────────────────────────────────────── #

FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CROP_SIZE=64x64
ENV PIXEL_RES=0.65

# Install system packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
    openjdk-17-jre-headless \
    ca-certificates \
    # Optional editors for debugging
    nano less vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies (pinned versions)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy==1.26.4 pandas==2.2.2 matplotlib==3.8.4 seaborn==0.13.2 scikit-image==0.22.0 imageio==2.34.1 tqdm==4.66.4 pyyaml==6.0.1 \
        gitpython==3.1.43 configargparse==1.7 tifffile==2024.5.22 opencv-python==4.9.0.80 typing-extensions==4.12.0 \
        torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        dill==0.3.8 \
        tensorboard==2.16.2

# Install Nextflow (latest)
RUN curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    chmod +x /usr/local/bin/nextflow

# Set workdir and copy code
WORKDIR /app
COPY . /app
COPY Workflow /app/Workflow
COPY TAP /app/TAP

# Ensure the runs directory exists and is always empty at build time
RUN mkdir -p /app/runs && rm -rf /app/runs/*

# Make helper scripts executable
RUN chmod +x /app/run_tap.sh

# Healthcheck for sanity (PyTorch import/version)
HEALTHCHECK CMD python -c "import torch; print(torch.__version__)" || exit 1

# Default shell
CMD ["bash"]
