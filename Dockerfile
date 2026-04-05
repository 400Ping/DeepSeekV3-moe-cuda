FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    bubblewrap

# Install Nsight Compute
RUN wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/2026_1_0/nsight_compute-linux-x86_64-2026.1.0.2.run && \
    chmod +x nsight_compute-linux-x86_64-2026.1.0.2.run && \
    ./nsight_compute-linux-x86_64-2026.1.0.2.run --noexec  --target /opt/nsight-compute && \
    ln -s /opt/nsight-compute/pkg/ncu /usr/local/bin/ncu

# Install Codex CLI
RUN npm i -g @openai/codex