# DLOps Assignment 5 – LoRA Fine-tuning & Adversarial Attacks

## Installation

### Prerequisites
- Python ≥ 3.9
- CUDA-capable GPU (recommended)
- Docker + NVIDIA Container Toolkit (for Docker-based execution)

### Install dependencies locally

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
pip install transformers>=4.38.0 peft>=0.10.0 huggingface_hub>=0.21.0
pip install optuna>=3.6.0
pip install wandb>=0.17.0
pip install adversarial-robustness-toolbox>=1.17.0
```

Or install everything at once from the requirements file:

```bash
pip install -r requirements.txt
```

### Authenticate services

```bash
# Weights & Biases (required for all scripts)
wandb login

# HuggingFace Hub (required only for Q1 Step 6: push model)
huggingface-cli login
```

---

## Docker

A `Dockerfile` is provided at the root of the project. It uses the official
`pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` base image so GPU training
works without any additional CUDA setup on the host.

### Requirements on the host machine
- [Docker Engine](https://docs.docker.com/engine/install/) ≥ 20.10
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU access inside the container)

### 1 · Build the image

```bash
# Run from the project root (where the Dockerfile lives)
docker build -t dlops-assg5 .
```

> The build installs all Python packages from `requirements.txt` and copies
> every `.py` script into `/workspace` inside the container.

---

### 2 · Run experiments

Pass your WandB API key via `-e WANDB_API_KEY=<your_key>`.  
Training data (CIFAR-10 / CIFAR-100) is downloaded automatically on first run;
mount a host directory to persist it across runs.

#### Q1 – Baseline (head-only, no LoRA)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_train.py --baseline
```

#### Q1 – Full LoRA grid (9 experiments)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_train.py
```

#### Q1 – Optuna hyperparameter search (20 trials)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_optuna.py
```

#### Q1 – Retrain best config and push to HuggingFace Hub

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -e HF_TOKEN=<your_huggingface_token> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q1_push.py
```

> Set `HF_REPO` inside `q1_push.py` to your HuggingFace repo name before
> building the image (or override it with `-e HF_REPO=<repo>`).

#### Q2(i) – FGSM Attack (Scratch vs IBM ART)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q2i_fgsm.py
```

#### Q2(ii) – Adversarial Detection (PGD & BIM)

```bash
docker run --gpus all --rm \
  -e WANDB_API_KEY=<your_wandb_key> \
  -v $(pwd)/data:/workspace/data \
  dlops-assg5 \
  python q2ii_detection.py
```

---

### 3 · Offline WandB mode

If you do not have network access inside the container, set:

```bash
-e WANDB_MODE=offline
```

Sync the run logs afterwards with:

```bash
wandb sync wandb/offline-run-*
```

---

### 4 · Saved model files

Model checkpoints are written to `/workspace/` inside the container.
Alternatively, mount the entire workspace:

```bash
-v $(pwd):/workspace
```

---

## Q1 – ViT-S LoRA on CIFAR-100

**Model:** `WinKawaks/vit-small-patch16-224` (ImageNet pretrained)  
**Dataset:** CIFAR-100  
**Task:** Fine-tune with LoRA injected into Q, K, V attention weights

### Files

| File | Purpose |
|------|---------|
| `q1_train.py` | Baseline + full LoRA grid (Rank ∈ {2,4,8} × Alpha ∈ {2,4,8}) |
| `q1_optuna.py` | Optuna HPO over rank, alpha, dropout (20 trials) |
| `q1_push.py` | Retrain best config and push to HuggingFace Hub |

### LoRA Configuration

| Setting | Value |
|---------|-------|
| Base model | WinKawaks/vit-small-patch16-224 |
| Target modules | query, key, value (Q, K, V) |
| Ranks tested | 2, 4, 8 |
| Alphas tested | 2, 4, 8 |
| Dropout | 0.1 |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR, T_max=10 |
| Epochs | 10 |
| Batch size | 128 |

### Results – LoRA Grid Search

| LoRA | Rank | Alpha | Dropout | Test Accuracy | Trainable Params |
|------|------|-------|---------|---------------|-----------------|
| No (Baseline) | – | – | – | 81.00% | 38,500 |
| Yes (Q,K,V) | 2 | 2 | 0.1 | **89.85%** | 93,796 |
| Yes (Q,K,V) | 2 | 4 | 0.1 | 89.73% | 93,796 |
| Yes (Q,K,V) | 2 | 8 | 0.1 | 89.32% | 93,796 |
| Yes (Q,K,V) | 4 | 2 | 0.1 | 89.84% | 149,092 |
| Yes (Q,K,V) | 4 | 4 | 0.1 | 89.62% | 149,092 |
| Yes (Q,K,V) | 4 | 8 | 0.1 | 89.63% | 149,092 |
| Yes (Q,K,V) | 8 | 2 | 0.1 | 89.91% | 259,684 |
| Yes (Q,K,V) | **8** | **4** | 0.1 | **90.21%** | 259,684 |
| Yes (Q,K,V) | 8 | 8 | 0.1 | 89.72% | 259,684 |

> **Best grid configuration:** Rank=8, Alpha=4 → **90.21% test accuracy**

### Results – Optuna HPO

| Setting | Value |
|---------|-------|
| Sampler | TPE |
| Trials | 20 |
| Best val accuracy | 89.86% |
| Best rank | 8 |
| Best alpha | 4 |
| Best dropout | 0.1271 |
| Final test accuracy (retrained) | **89.74%** |

### Detailed Per-Epoch Training Logs

#### Baseline – No LoRA (head-only fine-tuning)

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.7390 | 62.68% | 0.8900 | 77.26% |
| 2 | 0.7706 | 79.60% | 0.7360 | 79.52% |
| 3 | 0.6485 | 82.02% | 0.6850 | 80.70% |
| 4 | 0.5872 | 83.44% | 0.6594 | 80.86% |
| 5 | 0.5467 | 84.48% | 0.6476 | 81.34% |
| 6 | 0.5180 | 85.27% | 0.6387 | 81.26% |
| 7 | 0.4981 | 85.82% | 0.6344 | 81.44% |
| 8 | 0.4840 | 86.31% | 0.6299 | 81.56% |
| 9 | 0.4753 | 86.57% | 0.6297 | 81.52% |
| 10 | 0.4707 | 86.69% | 0.6291 | **81.48%** |

#### Experiment 1 – LoRA Rank=2, Alpha=2

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2713 | 72.05% | 0.4334 | 87.66% |
| 2 | 0.3708 | 88.69% | 0.3688 | 88.66% |
| 3 | 0.2969 | 90.79% | 0.3458 | 89.48% |
| 4 | 0.2524 | 92.20% | 0.3378 | 89.50% |
| 5 | 0.2202 | 93.15% | 0.3357 | 89.76% |
| 6 | 0.1967 | 94.02% | 0.3350 | 89.66% |
| 7 | 0.1795 | 94.65% | 0.3302 | 89.98% |
| 8 | 0.1673 | 95.08% | 0.3305 | 90.04% |
| 9 | 0.1599 | 95.39% | 0.3302 | 90.02% |
| 10 | 0.1558 | 95.55% | 0.3302 | **89.98%** |

#### Experiment 2 – LoRA Rank=2, Alpha=4

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2163 | 72.87% | 0.4252 | 87.28% |
| 2 | 0.3625 | 89.04% | 0.3697 | 88.88% |
| 3 | 0.2870 | 91.15% | 0.3440 | 89.24% |
| 4 | 0.2415 | 92.57% | 0.3409 | 89.52% |
| 5 | 0.2075 | 93.75% | 0.3351 | 89.56% |
| 6 | 0.1823 | 94.58% | 0.3340 | 89.90% |
| 7 | 0.1638 | 95.30% | 0.3364 | 89.54% |
| 8 | 0.1511 | 95.80% | 0.3382 | 89.66% |
| 9 | 0.1433 | 96.09% | 0.3374 | 89.88% |
| 10 | 0.1386 | 96.31% | 0.3372 | **89.92%** |

#### Experiment 3 – LoRA Rank=2, Alpha=8

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.1697 | 73.77% | 0.4263 | 87.40% |
| 2 | 0.3502 | 89.44% | 0.3773 | 88.84% |
| 3 | 0.2742 | 91.58% | 0.3627 | 89.18% |
| 4 | 0.2258 | 93.16% | 0.3532 | 89.20% |
| 5 | 0.1903 | 94.31% | 0.3545 | 89.34% |
| 6 | 0.1635 | 95.26% | 0.3575 | 89.40% |
| 7 | 0.1437 | 95.99% | 0.3620 | 89.10% |
| 8 | 0.1306 | 96.47% | 0.3615 | 89.24% |
| 9 | 0.1218 | 96.84% | 0.3633 | 89.32% |
| 10 | 0.1170 | 97.08% | 0.3629 | **89.32%** |

#### Experiment 4 – LoRA Rank=4, Alpha=2

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2544 | 72.36% | 0.4340 | 87.20% |
| 2 | 0.3682 | 88.93% | 0.3682 | 88.66% |
| 3 | 0.2939 | 90.82% | 0.3496 | 89.22% |
| 4 | 0.2501 | 92.26% | 0.3358 | 89.62% |
| 5 | 0.2181 | 93.36% | 0.3327 | 89.36% |
| 6 | 0.1940 | 94.07% | 0.3303 | 89.72% |
| 7 | 0.1771 | 94.72% | 0.3281 | 89.78% |
| 8 | 0.1651 | 95.16% | 0.3302 | 89.82% |
| 9 | 0.1573 | 95.49% | 0.3292 | 89.92% |
| 10 | 0.1534 | 95.68% | 0.3289 | **89.96%** |

#### Experiment 5 – LoRA Rank=4, Alpha=4

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2151 | 72.75% | 0.4265 | 87.40% |
| 2 | 0.3561 | 89.29% | 0.3632 | 88.70% |
| 3 | 0.2812 | 91.38% | 0.3461 | 89.34% |
| 4 | 0.2349 | 92.90% | 0.3395 | 89.36% |
| 5 | 0.2006 | 94.00% | 0.3342 | 89.52% |
| 6 | 0.1745 | 94.93% | 0.3346 | 89.62% |
| 7 | 0.1557 | 95.68% | 0.3331 | 89.84% |
| 8 | 0.1431 | 96.07% | 0.3359 | 89.76% |
| 9 | 0.1351 | 96.38% | 0.3352 | 89.72% |
| 10 | 0.1308 | 96.55% | 0.3355 | **89.78%** |

#### Experiment 6 – LoRA Rank=4, Alpha=8

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.1825 | 73.57% | 0.4210 | 88.12% |
| 2 | 0.3498 | 89.50% | 0.3541 | 89.16% |
| 3 | 0.2698 | 91.61% | 0.3402 | 89.70% |
| 4 | 0.2193 | 93.32% | 0.3368 | 89.62% |
| 5 | 0.1804 | 94.56% | 0.3331 | 89.46% |
| 6 | 0.1517 | 95.69% | 0.3309 | 89.52% |
| 7 | 0.1313 | 96.44% | 0.3318 | 89.86% |
| 8 | 0.1179 | 96.96% | 0.3346 | 89.74% |
| 9 | 0.1095 | 97.20% | 0.3348 | 89.78% |
| 10 | 0.1050 | 97.45% | 0.3355 | **89.90%** |

#### Experiment 7 – LoRA Rank=8, Alpha=2

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2727 | 72.12% | 0.4410 | 87.20% |
| 2 | 0.3687 | 88.94% | 0.3720 | 88.80% |
| 3 | 0.2939 | 91.02% | 0.3499 | 89.18% |
| 4 | 0.2484 | 92.37% | 0.3401 | 89.44% |
| 5 | 0.2165 | 93.44% | 0.3313 | 89.74% |
| 6 | 0.1926 | 94.12% | 0.3291 | 89.90% |
| 7 | 0.1746 | 94.86% | 0.3323 | 89.92% |
| 8 | 0.1626 | 95.32% | 0.3305 | 89.78% |
| 9 | 0.1549 | 95.60% | 0.3304 | 89.80% |
| 10 | 0.1510 | 95.74% | 0.3305 | **89.82%** |

#### Experiment 8 – LoRA Rank=8, Alpha=4

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.2349 | 72.54% | 0.4321 | 87.34% |
| 2 | 0.3554 | 89.25% | 0.3686 | 88.76% |
| 3 | 0.2800 | 91.34% | 0.3434 | 89.48% |
| 4 | 0.2324 | 92.89% | 0.3408 | 89.56% |
| 5 | 0.1971 | 94.09% | 0.3393 | 89.46% |
| 6 | 0.1710 | 94.92% | 0.3381 | 89.86% |
| 7 | 0.1522 | 95.67% | 0.3370 | 89.78% |
| 8 | 0.1392 | 96.17% | 0.3392 | 89.80% |
| 9 | 0.1310 | 96.49% | 0.3378 | 89.84% |
| 10 | 0.1268 | 96.62% | 0.3377 | **90.02%** |

#### Experiment 9 – LoRA Rank=8, Alpha=8

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|----------:|---------------:|---------:|-------------:|
| 1 | 1.1813 | 73.52% | 0.4277 | 87.12% |
| 2 | 0.3461 | 89.62% | 0.3656 | 88.54% |
| 3 | 0.2654 | 91.86% | 0.3518 | 89.56% |
| 4 | 0.2131 | 93.54% | 0.3450 | 89.14% |
| 5 | 0.1747 | 94.74% | 0.3393 | 89.86% |
| 6 | 0.1461 | 95.80% | 0.3404 | 89.48% |
| 7 | 0.1260 | 96.52% | 0.3464 | 89.58% |
| 8 | 0.1119 | 97.08% | 0.3439 | 89.88% |
| 9 | 0.1037 | 97.35% | 0.3460 | 89.58% |
| 10 | 0.0991 | 97.59% | 0.3463 | **89.66%** |

---

## Q2(i) – FGSM Attack: Scratch vs IBM ART

**Model:** ResNet-18 trained from scratch on CIFAR-10 (target ≥ 72% clean accuracy)  
**Dataset:** CIFAR-10  
**Attacks:** FGSM implemented manually + FGSM via IBM ART

### File

| File | Purpose |
|------|---------|
| `q2i_fgsm.py` | Train ResNet-18, apply FGSM (scratch + ART), log results to WandB |


This will:
1. Train ResNet-18 from scratch for 30 epochs (SGD + CosineAnnealingLR).
2. Run FGSM from scratch at ε ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.30}.
3. Run FGSM via IBM ART at the same ε values.
4. Save and log a visual comparison grid (`fgsm_comparison.png`) to WandB.
5. Log an accuracy table to WandB.

### Training Configuration

| Setting | Value |
|---------|-------|
| Model | ResNet-18 (from scratch, no pretrained weights) |
| Dataset | CIFAR-10 (train: 50k, test: 10k) |
| Epochs | 30 |
| Optimizer | SGD, lr=0.1, momentum=0.9, weight_decay=5e-4 |
| Scheduler | CosineAnnealingLR, T_max=30 |
| Batch size | 256 |
| Data augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip |

### Results – Accuracy vs Perturbation Strength

**Baseline clean accuracy:** 84.20%

| ε (epsilon) | FGSM Scratch Acc | FGSM ART Acc | Scratch Drop | ART Drop |
|-------------|-----------------|--------------|-------------|---------|
| 0.010 | 70.04% | 82.53% | 14.16 pp | 1.67 pp |
| 0.020 | 55.70% | 80.10% | 28.50 pp | 4.10 pp |
| 0.050 | 26.05% | 70.91% | 58.15 pp | 13.29 pp |
| 0.100 | 9.82% | 56.49% | 74.38 pp | 27.71 pp |
| 0.200 | 4.48% | 38.33% | 79.72 pp | 45.87 pp |
| 0.300 | 3.67% | 28.78% | **80.53 pp** | **55.42 pp** |

---

## Q2(ii) – Adversarial Detection Model

**Victim model:** ResNet-18 trained from scratch on CIFAR-10  
**Detector model:** ResNet-34 (binary: 0=clean, 1=adversarial)  
**Attacks:** PGD and BIM via IBM ART

### File

| File | Purpose |
|------|---------|
| `q2ii_detection.py` | Train victim, generate PGD/BIM examples, train ResNet-34 detectors, log to WandB |

This will:
1. Train a ResNet-18 victim model (≥72% target accuracy).
2. Generate adversarial examples using PGD and BIM via IBM ART.
3. Generate FGSM examples (scratch + ART) for WandB sample logging.
4. Train two independent ResNet-34 binary detectors (one per attack type).
5. Report and log detection accuracy to WandB.

### Configuration

| Setting | Value |
|---------|-------|
| Victim model | ResNet-18 (from scratch) |
| Detector model | ResNet-34 (binary output: 0=clean, 1=adversarial) |
| Dataset | CIFAR-10 (10,000 test images used for attack generation) |
| PGD: ε | 8/255, step=2/255, iterations=40 |
| BIM: ε | 8/255, step=2/255, iterations=40 |
| FGSM: ε | 8/255 |
| Detector optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| Detector scheduler | CosineAnnealingLR, T_max=20 |
| Detector training | 20 epochs with val-accuracy-based best model selection |
| Batch size | 128 |

### Results – Victim Model

| Metric | Value |
|--------|-------|
| Clean test accuracy | **85.45%** |

### Results – Adversarial Detection

| Attack | Method | Detection Accuracy |
|--------|--------|-------------------|
| PGD | IBM ART | **99.27%** |
| BIM | IBM ART | **99.15%** |

Both detectors **exceed the 70% detection accuracy requirement** by a large margin.

---

## WandB Logging Summary

| Script | Project | What is Logged |
|--------|---------|----------------|
| `q1_train.py` | `ViT-S-Finetune` | Train/val loss & acc, class-wise test histogram, LoRA gradient norms, trainable param counts |
| `q1_optuna.py` | `ViT-S-Finetune` | Per-trial val accuracy, best hyperparameters |
| `q2i_fgsm.py` | `FGSM-Attack` | Clean accuracy, ε vs scratch/ART accuracy table, visual comparison grid |
| `q2ii_detection.py` | `Adversarial-Detection` | Detector val accuracy per epoch, detection summary table, 10-sample image strips |

---

## Links

### WandB Links

- https://wandb.ai/m25csa002-iit-jodhpur/ViT-S-Finetune?nw=nwuserm25csa002
- https://wandb.ai/m25csa002-iit-jodhpur/ViT-S-Finetune-optuna?nw=nwuserm25csa002
- https://wandb.ai/m25csa002-iit-jodhpur/FGSM-Attack?nw=nwuserm25csa002
- https://wandb.ai/m25csa002-iit-jodhpur/Adversarial-Detection?nw=nwuserm25csa002

### HuggingFace Link

- https://huggingface.co/AdiSingh003/vit-small-lora-cifar100/tree/main