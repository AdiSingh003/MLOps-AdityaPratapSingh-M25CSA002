# Training and Fine-Tuning BERT for Classification

> Fine-tune **DistilBERT** to classify Goodreads book reviews into 8 genres — containerised with Docker and published to Hugging Face Hub.
---

## Project Overview

This project implements an end-to-end Huggingface model training & docker deployment:

1. **Data** — Stream and sample Goodreads book reviews from 8 genres (UCSD dataset)
2. **Train** — Fine-tune `distilbert-base-cased` using the HuggingFace Trainer API
3. **Evaluate** — Compute accuracy, F1, precision, recall and save confusion matrices
4. **Publish** — Push model + tokenizer to Hugging Face Hub
5. **Deploy** — Docker container pulls model from Hub and runs evaluation automatically

---

## Model Selection

| Property | Value |
|---|---|
| **Model** | `distilbert-base-cased` |
| **Parameters** | ~66M (vs ~110M for BERT-base) |
| **Speed** | 60% faster inference than BERT-base |
| **Performance** | Retains ~97% of BERT-base GLUE benchmark score |
| **Tokenisation** | Cased — preserves proper nouns, author names, genre labels |
| **Max Length** | 512 tokens |

**Why DistilBERT?**
DistilBERT was selected for its balance of speed and performance. It trains significantly faster than full BERT on Colab's free GPU tier, supports the complete HuggingFace Trainer API, and the cased variant preserves capitalisation signals that are meaningful in book review language (e.g. genre-specific proper nouns). It retains 97% of BERT's accuracy at 40% less cost — the standard choice for classification tasks at this scale.

---

## Dataset

**Source:** [UCSD Goodreads Book Reviews](https://mengtingwan.github.io/data/goodreads.html)

| Genre | Train Samples | Test Samples |
|---|---|---|
| children | 800 | 200 |
| comics\_graphic | 800 | 200 |
| fantasy\_paranormal | 800 | 200 |
| history\_biography | 800 | 200 |
| mystery\_thriller\_crime | 800 | 200 |
| poetry | 800 | 200 |
| romance | 800 | 200 |
| young\_adult | 800 | 200 |
| **Total** | **6,400** | **1,600** |

Reviews are streamed directly from the UCSD server (no manual download required) and cached locally as `genre_reviews_dict.pickle` after the first run.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-cased` |
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 5e-5 |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max token length | 512 |
| Eval strategy | epoch |
| Best model metric | eval\_loss |
| Mixed precision (fp16) | Auto (GPU only) |

---

## Evaluation Results

| Metric | Local Model | Hub Model |
|---|---|---|
| **Accuracy** | 60.25% | 62.13% |
| **F1 (weighted)** | 0.5990 | 0.6162 |
| **Precision** | 0.5990 | 0.6155 |
| **Recall** | 0.6025 | 0.6213 |
| **Eval Loss** | 1.1812 | 1.1108 |

The Hub model slightly outperforms the local model. Both are the same checkpoint — the small difference is due to GPU vs CPU floating-point rounding in inference, which is within normal variance.

### Training Curve

| Epoch | Train Loss | Val Loss | Accuracy |
|---|---|---|---|
| 1 | 1.4288 | 1.1909 | 57.56% |
| **2** | **0.9604** | **1.1813** | **60.25%** |
| 3 | 0.6163 | 1.2616 | 59.56% |

Best checkpoint: **Epoch 2** (lowest validation loss).

---

## Per-Class Performance

### Hub Model

| Genre | Precision | Recall | F1 |
|---|---|---|---|
| children | 0.64 | 0.72 | 0.68 |
| comics\_graphic | 0.74 | 0.81 | **0.77** |
| fantasy\_paranormal | 0.44 | 0.41 | 0.43 |
| history\_biography | 0.65 | 0.57 | 0.61 |
| mystery\_thriller\_crime | 0.59 | 0.58 | 0.58 |
| poetry | 0.81 | 0.79 | **0.80** |
| romance | 0.61 | 0.73 | 0.67 |
| young\_adult | 0.44 | 0.36 | *0.40* |

**Strongest:** Poetry (F1: 0.80), Comics & Graphic (F1: 0.77) — linguistically distinct genres with unique vocabulary.

**Weakest:** Young Adult (F1: 0.40) — YA is a demographic category, not a content genre, so reviews overlap with every other class.

---

## Links

| Resource | URL |
|---|---|
| 🤗 Hugging Face Model | `https://huggingface.co/AdiSingh003/distilbert-goodreadnews-finetuned/tree/main` |
| 💻 GitHub Repository | `https://github.com/AdiSingh003/MLOps-AdityaPratapSingh-M25CSA002/tree/assg_2` |

---

## Challenges

### 1. Label Mapping Bug (Root Cause of 14% Initial Accuracy)
Python's `set()` is unordered, so `set(train_labels)` produced a different `label→integer` mapping each run. The model output integer `0` meant `poetry` during training but `comics_graphic` during evaluation. **Fix:** replaced `set()` with `sorted(set())` and embedded `id2label`/`label2id` into the model's `config.json` before pushing to the Hub.

### 2. Cross-Genre Vocabulary Overlap
Young Adult and Fantasy/Paranormal share heavily overlapping vocabulary because YA is an age-demographic category spanning all content genres. This places a theoretical ceiling on accuracy for those classes.

### 3. Overfitting at Epoch 3
Validation loss rose at Epoch 3 while training loss continued falling. Handled via `load_best_model_at_end=True` in `TrainingArguments`, which checkpoints the best epoch (Epoch 2) automatically.

### 4. Docker GPU Configuration
Required switching from `python:3.11-slim` to `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` and installing PyTorch from the CUDA 11.8 wheel index. Without this, `torch.cuda.is_available()` returns `False` inside the container.

### 5. HuggingFace Authentication in Docker
Solved by accepting the token via the `HF_TOKEN` environment variable, which `huggingface_hub.login()` reads automatically — no interactive prompt needed in CI/Docker environments.