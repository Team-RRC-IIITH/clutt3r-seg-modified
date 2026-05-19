FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    DUODUOCLIP_ROOT=/opt/DuoduoCLIP

WORKDIR /workspace

ARG INSTALL_DUODUOCLIP=0
ARG DUODUOCLIP_REPO=https://github.com/3dlg-hcvc/DuoduoCLIP.git
ARG DUODUOCLIP_REF=main

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md LICENSE THIRD_PARTY.md ./
COPY clutt3rseg ./clutt3rseg
COPY scripts ./scripts
COPY initial_segmenter.sh update_segmenter.sh ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e .

RUN if [ "${INSTALL_DUODUOCLIP}" = "1" ]; then \
        git clone "${DUODUOCLIP_REPO}" "${DUODUOCLIP_ROOT}" \
        && git -C "${DUODUOCLIP_ROOT}" checkout "${DUODUOCLIP_REF}" \
        && python -m pip install -r "${DUODUOCLIP_ROOT}/requirements.txt" \
        && python -m pip install -e "${DUODUOCLIP_ROOT}/open_clip_mod"; \
    fi

RUN python -m compileall -q clutt3rseg

CMD ["bash"]
