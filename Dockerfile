FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY q1_train.py      .
COPY q1_optuna.py     .
COPY q1_push.py       .
COPY q2i_fgsm.py      .
COPY q2ii_detection.py .

RUN mkdir -p /workspace/data

ENV WANDB_API_KEY=""
ENV HF_TOKEN=""
ENV WANDB_MODE="online"

CMD ["python", "-c", "\
print('=== DLOps Assignment 5 Docker Image ==='); \
print(''); \
print('Q1 – ViT-S LoRA on CIFAR-100:'); \
print('  Baseline:  python q1_train.py --baseline'); \
print('  LoRA grid: python q1_train.py'); \
print('  Optuna:    python q1_optuna.py'); \
print('  HF Push:   python q1_push.py'); \
print(''); \
print('Q2(i) – FGSM:'); \
print('  python q2i_fgsm.py'); \
print(''); \
print('Q2(ii) – Adversarial Detection:'); \
print('  python q2ii_detection.py'); \
print(''); \
print('Pass your WandB API key: docker run -e WANDB_API_KEY=<key> ...'); \
"]