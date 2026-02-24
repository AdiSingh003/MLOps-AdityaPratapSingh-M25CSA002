import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


def build_label_maps(labels: list) -> tuple:
    unique = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(labels: list, label2id: dict) -> list:
    return [label2id[y] for y in labels]


def decode_labels(label_ids: list, id2label: dict) -> list:
    return [id2label[i] for i in label_ids]


def compute_metrics(pred) -> dict:
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
      labels,
      preds,
      average="weighted"
  )

    return {
        "accuracy":acc,
        "f1":f1,
        "precision":precision,
        "recall":recall,
    }


def save_confusion_matrix(
    true_labels:      list,
    predicted_labels: list,
    label_names:      list,
    path:             str  = "./results/confusion_matrix.png",
    normalize:        bool = False,
) -> str:
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cm = confusion_matrix(true_labels, predicted_labels, labels=label_names)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
        title = "Confusion Matrix (normalised)"
    else:
        fmt, vmax = "d", None
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Purples",
        xticklabels=label_names,
        yticklabels=label_names,
        linewidths=0.5,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def save_results(results: dict, path: str = "./results/eval_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def load_results(path: str = "./results/eval_results.json") -> dict:
    with open(path) as fh:
        return json.load(fh)


def print_classification_report(true_labels, predicted_labels,save_path) -> str:
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    print(report)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)