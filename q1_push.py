"""
Q1 Step 6: Retrain with best Optuna hyperparameters and push to HuggingFace.
Also saves weights locally for GitHub upload.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi
from tqdm import tqdm

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 10
BATCH_SIZE  = 128
LR          = 3e-4
NUM_CLASSES = 100
DATA_DIR    = "data"
HF_REPO     = "AdiSingh003/vit-small-lora-cifar100"   # <-- change this
TARGET_MODULES = ["query", "key", "value"]


def get_loaders():
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
    val_ds.dataset.transform = val_tf
    return (
        DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True),
        DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
    )


def build_model(rank, alpha, dropout):
    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    cfg   = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                       target_modules=TARGET_MODULES, bias="none")
    model = get_peft_model(base, cfg)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return model


def retrain_and_push(rank: int, alpha: int, dropout: float):
    print(f"\n=== Retraining best model: rank={rank}, alpha={alpha}, dropout={dropout} ===")
    train_loader, val_loader, test_loader = get_loaders()
    model = build_model(rank, alpha, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(imgs).logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        print(f"  Epoch {epoch:02d} | Val Acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lora_model.pt")

    # ── Test ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load("best_lora_model.pt"))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    print(f"\nFinal Test Accuracy: {correct/total:.4f}")

    # ── Save full PEFT model ───────────────────────────────────────────────────
    os.makedirs("best_model_hf", exist_ok=True)
    model.save_pretrained("best_model_hf")
    print("Saved PEFT model to ./best_model_hf/")

    # ── Push to HuggingFace ───────────────────────────────────────────────────
    # Make sure you run: huggingface-cli login
    try:
        api = HfApi()
        api.create_repo(repo_id=HF_REPO, exist_ok=True)
        api.upload_folder(folder_path="best_model_hf", repo_id=HF_REPO)
        print(f"Model pushed to HuggingFace: https://huggingface.co/{HF_REPO}")
    except Exception as e:
        print(f"HF push failed (set HF_REPO and login): {e}")
        print("Weights saved locally at best_lora_model.pt and best_model_hf/")


if __name__ == "__main__":
    # Example: call with manually specified best params
    retrain_and_push(rank=4, alpha=8, dropout=0.1)
