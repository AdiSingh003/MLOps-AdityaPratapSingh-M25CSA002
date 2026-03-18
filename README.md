# Assignment 4 — English → Hindi Transformer Translation
## Hyperparameter Optimisation with Ray Tune + Optuna
---

## Overview

This project trains a **from-scratch PyTorch Transformer** for English-to-Hindi neural machine translation, then uses **Ray Tune** paired with **Optuna's TPE search algorithm** and the **ASHA early-stopping scheduler** to find a better hyperparameter configuration — achieving a higher BLEU score in significantly fewer epochs than the hardcoded baseline.

| | Baseline | Ray Tune + Optuna |
|---|---|---|
| Epochs | 100 | **30** |
| Training time | ~95 min | **14.9 min** (retrain only) |
| Final loss | 0.0979 | 0.2733 |
| **BLEU Score** | 53.95 | **65.61 (+11.66)** |

---

## Requirements

### Python Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install ray[tune] optuna pandas seaborn matplotlib tqdm nltk psutil
```

### Environment

| Dependency | Tested Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| Ray | 2.54.0 |
| Optuna | 3.x |
| CUDA | 12.8 |

> **Conda users:** `conda activate <env_name>`

---

## Dataset

The dataset is the **English–Hindi parallel corpus** from Tatoeba, hosted on Google Drive.

| Property | Value |
|---|---|
| Source | Tatoeba (English–Hindi pairs) |
| Total pairs | 13,186 |
| Avg. English length | 5.6 words |
| Avg. Hindi length | 6.3 words |
| English vocab size | 4,117 tokens |
| Hindi vocab size | 4,044 tokens |

The script downloads the dataset automatically at runtime via:
```python
file_id = '1dPWcMzr0H5utKjqa-HUod1_QhwXSsJfI'
url = f"https://drive.google.com/uc?id={file_id}"
```

---

## Model Architecture

A standard Transformer (Vaswani et al., 2017) implemented from scratch in PyTorch:

| Component | Details |
|---|---|
| Positional Encoding | Sinusoidal, max_len=5000 |
| Multi-Head Attention | Scaled dot-product, dropout on attention weights |
| Feed-Forward Network | Linear → ReLU → Dropout → Linear |
| Layer Normalisation | Custom implementation (γ, β parameters) |
| `d_model` | **512** (fixed across all experiments) |
| Loss | CrossEntropyLoss (ignores `<pad>` index) |
| Optimiser | Adam |

`d_model = 512` is fixed to ensure `d_model % num_heads == 0` holds for all candidate values of `num_heads` in the search space.

---

## How to Run

### Option 1 — SLURM (HPC cluster)

```bash
sbatch en_to_hi_slurm.sh
```

### Option 2 — Local / interactive

```bash
python m25csa002_ass_4_tuned_en_to_hi.py
```

> On a machine without a GPU the script falls back to CPU automatically. Expect significantly longer runtimes.

---

## Script Walkthrough

The script runs end-to-end in a single execution:

### Part 1 — Baseline Training
1. Loads and preprocesses the dataset
2. Builds English and Hindi vocabularies (`freq_threshold=2`)
3. Trains a Transformer for **100 epochs** with hardcoded hyperparameters
4. Saves weights to `transformer_translation_final.pth`
5. Evaluates BLEU on a 5-sentence validation set

### Part 2 — Ray Tune + Optuna Sweep
6. Defines a `train_tune(config)` function compatible with Ray Tune
   - All imports, vocab building, and model construction are **self-contained inside the function** (required because Ray workers run in isolated processes)
7. Defines the **6-hyperparameter search space** (see table below)
8. Initialises Ray, respecting SLURM CPU/GPU allocations via environment variables
9. Runs **20 trials × ≤30 epochs** with Optuna TPE + ASHA pruning
10. Extracts the best configuration

### Part 3 — Best Config Retrain & Evaluation
11. Retrains a fresh model using the best config for 30 epochs
12. Evaluates BLEU score and prints translation samples
13. Saves all artefacts and plots

---

## Hyperparameter Search Space

| Hyperparameter | API | Range / Choices | Rationale |
|---|---|---|---|
| Learning rate | `tune.loguniform` | [1e-5, 1e-3] | Log scale; most impactful single param |
| Batch size | `tune.choice` | {32, 64, 128} | Affects gradient noise & convergence speed |
| Num. attention heads | `tune.choice` | {4, 8} | Both divide 512; controls attention granularity |
| Feed-forward dim (d_ff) | `tune.choice` | {1024, 2048, 4096} | Controls per-layer capacity |
| Dropout rate | `tune.uniform` | [0.05, 0.40] | Primary regularisation knob |
| Num. layers | `tune.choice` | {4, 6} | Shallower = faster; deeper = more expressive |

### Search Algorithm Configuration

```python
optuna_search  = OptunaSearch(metric="loss", mode="min")

asha_scheduler = ASHAScheduler(
    metric="loss", mode="min",
    max_t=30,            # max epochs per trial
    grace_period=5,      # min epochs before pruning
    reduction_factor=2   # keep top 50% at each rung
)

tuner = tune.Tuner(
    tune.with_resources(train_tune, resources={"cpu": 1, "gpu": 0.5}),
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        scheduler=asha_scheduler,
        num_samples=20,
    ),
    param_space=search_space,
)
```

---

## Best Configuration

Found by Optuna after 16 completed trials (trial `train_tune_de2cf572`):

```json
{
  "lr": 0.00016064935184656424,
  "batch_size": 128,
  "num_heads": 8,
  "d_ff": 2048,
  "dropout": 0.13640224510542814,
  "num_layers": 6,
  "num_epochs": 30
}
```

Stored in `best_hyperparams.json`.

---

## Memory & GPU Configuration

The script is tuned for HPC SLURM environments where Ray may otherwise over-subscribe system resources:

```python
# Reads SLURM environment variables to stay within allocation
slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
slurm_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "1" if cuda else "0"))
MAX_PARALLEL = min(slurm_cpus, 2)   # 2 trials × ~9 GB each fits in 23.5 GB GPU

ray.init(
    num_cpus=MAX_PARALLEL,
    num_gpus=slurm_gpus,
    object_store_memory=object_store_mb * 1024 * 1024,  # 15% of total RAM
)
```

Also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce GPU memory fragmentation during `backward()`.

---

## Output Files

| File | Description |
|---|---|
| `checkpoint.pt` | Resumable checkpoint saved every epoch (baseline) |
| `transformer_translation_final.pth` | Baseline model state dict |
| `m25csa002_ass_4_best_model.pth` | Best tuned model state dict |
| `en_vocab.pkl` | Serialised English `Vocabulary` object |
| `hi_vocab.pkl` | Serialised Hindi `Vocabulary` object |
| `best_hyperparams.json` | Best config from Optuna sweep |
| `plots/` | All 6 saved visualisation PNGs |

---

## Loading the Best Model

```python
import torch, pickle
from m25csa002_ass_4_tuned_en_to_hi import Transformer, encode_sentence

# Load vocabularies
with open("en_vocab.pkl", "rb") as f: en_vocab = pickle.load(f)
with open("hi_vocab.pkl", "rb") as f: hi_vocab = pickle.load(f)

# Reconstruct model with best hyperparameters
model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=512, num_layers=6, num_heads=8,
    d_ff=2048, max_len=50, dropout=0.136
)
model.load_state_dict(torch.load("m25csa002_ass_4_best_model.pth", map_location="cpu"))
model.eval()

# Translate
from m25csa002_ass_4_tuned_en_to_hi import translate_sentence
print(translate_sentence(model, "How are you?", en_vocab, hi_vocab))
# → आप कैसी हैं?
```

---

## BLEU Evaluation

BLEU is computed using NLTK's `corpus_bleu` with `SmoothingFunction.method4` on a fixed 5-sentence validation set:

```python
VAL_DATASET = [
    ("I love you.",                 "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?",                "आप कैसे हैं?"),
    ("You should sleep.",           "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.",            "मुझे टॉम को बताने दीजिए।"),
]

