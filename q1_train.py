"""
Q1: ViT-S LoRA Fine-tuning on CIFAR-100
- Baseline: finetune classification head only
- LoRA: inject into Q, K, V attention weights using PEFT
- Ranks: 2, 4, 8 | Alpha: 2, 4, 8 | Dropout: 0.1
- Logs to WandB
"""

import os
import itertools
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 3e-4
NUM_CLASSES = 100
DATA_DIR    = "data"
WANDB_PROJECT = "ViT-S-Finetune"

RANKS    = [2, 4, 8]
ALPHAS   = [2, 4, 8]
DROPOUT  = 0.1
TARGET_MODULES = ["query", "key", "value"]   # Q, K, V in ViT attention

# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds    = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=val_tf)

    val_size   = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    # Use val transform for val subset
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, test_ds.classes

# ── Model helpers ─────────────────────────────────────────────────────────────
def build_baseline_model():
    """ViT-S pretrained on ImageNet, only classifier head trainable."""
    model = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model

def build_lora_model(rank: int, alpha: int, dropout: float):
    """ViT-S with LoRA injected into Q, K, V + trainable head."""
    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    # Also keep the classifier head trainable
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return model

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs).logits
            loss    = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, correct / total

# ── Gradient logger for LoRA weights ─────────────────────────────────────────
def log_lora_grads(model, step):
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name and param.grad is not None:
            grad_stats[f"grad_norm/{name}"] = param.grad.norm().item()
    wandb.log(grad_stats, step=step)

# ── Per-class test accuracy ───────────────────────────────────────────────────
def class_accuracy(model, loader, classes):
    model.eval()
    correct = torch.zeros(NUM_CLASSES)
    totals  = torch.zeros(NUM_CLASSES)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).logits.argmax(1)
            for c in range(NUM_CLASSES):
                mask = labels == c
                correct[c] += (preds[mask] == c).sum().item()
                totals[c]  += mask.sum().item()
    acc_per_class = (correct / totals.clamp(min=1)).tolist()
    return {classes[i]: acc_per_class[i] for i in range(NUM_CLASSES)}

# ── Single experiment ─────────────────────────────────────────────────────────
def train_experiment(model, exp_name: str, config: dict, train_loader, val_loader, test_loader, classes):
    wandb.init(project=WANDB_PROJECT, name=exp_name, config=config, reinit=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.to(DEVICE)
    best_val_acc = 0.0
    global_step  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        # Log LoRA grad norms after each epoch
        if config.get("lora"):
            log_lora_grads(model, step=epoch)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, train=False)
        scheduler.step()

        print(f"[{exp_name}] Epoch {epoch:02d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss":   val_loss, "val/acc":   val_acc,
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{exp_name}.pt")

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"best_{exp_name}.pt"))
    _, test_acc = run_epoch(model, test_loader, criterion, None, train=False)
    cls_acc     = class_accuracy(model, test_loader, classes)

    # Class-wise histogram
    wandb.log({
        "test/overall_acc":  test_acc,
        "test/class_acc_bar": wandb.plot.bar(
            wandb.Table(data=[[k, v] for k, v in cls_acc.items()],
                        columns=["class", "accuracy"]),
            "class", "accuracy", title="Class-wise Test Accuracy"
        ),
        "trainable_params": count_trainable(model),
    })

    wandb.finish()
    return test_acc, count_trainable(model)

# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    train_loader, val_loader, test_loader, classes = get_loaders()
    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.baseline:
        print("\n=== Baseline (no LoRA) ===")
        model  = build_baseline_model()
        config = {"lora": False, "rank": None, "alpha": None, "dropout": None}
        test_acc, n_params = train_experiment(
            model, "baseline_no_lora", config, train_loader, val_loader, test_loader, classes
        )
        results.append({
            "LoRA layers": "without", "Rank": "-", "Alpha": "-",
            "Dropout": "-", "Overall Test Accuracy": f"{test_acc:.4f}",
            "Trainable Params": n_params
        })

    # ── LoRA grid ─────────────────────────────────────────────────────────────
    exp_no = 1
    for rank, alpha in itertools.product(RANKS, ALPHAS):
        exp_name = f"lora_r{rank}_a{alpha}"
        print(f"\n=== Experiment {exp_no}: Rank={rank}, Alpha={alpha} ===")
        config = {"lora": True, "rank": rank, "alpha": alpha, "dropout": DROPOUT,
                  "target_modules": TARGET_MODULES}
        model  = build_lora_model(rank, alpha, DROPOUT)
        test_acc, n_params = train_experiment(
            model, exp_name, config, train_loader, val_loader, test_loader, classes
        )
        results.append({
            "LoRA layers": "with (Q,K,V)", "Rank": rank, "Alpha": alpha,
            "Dropout": DROPOUT, "Overall Test Accuracy": f"{test_acc:.4f}",
            "Trainable Params": n_params
        })
        exp_no += 1

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n\n=== Summary Table ===")
    print(f"{'LoRA layers':<20} {'Rank':<6} {'Alpha':<7} {'Dropout':<9} "
          f"{'Test Acc':<12} {'Trainable Params'}")
    for r in results:
        print(f"{r['LoRA layers']:<20} {str(r['Rank']):<6} {str(r['Alpha']):<7} "
              f"{str(r['Dropout']):<9} {r['Overall Test Accuracy']:<12} {r['Trainable Params']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no LoRA)")
    args = parser.parse_args()
    main(args)
