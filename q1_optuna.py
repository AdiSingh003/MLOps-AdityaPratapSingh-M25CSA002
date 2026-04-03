"""
Q1 Step 5: Optuna HPO for LoRA hyperparameters on CIFAR-100 ViT-S
Searches: rank in {2,4,8}, alpha in {2,4,8}, dropout in [0.05, 0.2]
"""

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import wandb

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS_HPO  = 5          # shorter runs during search
BATCH_SIZE  = 128
LR          = 3e-4
NUM_CLASSES = 100
DATA_DIR    = "data"
N_TRIALS    = 20
WANDB_PROJECT = "ViT-S-Finetune-optuna"

TARGET_MODULES = ["query", "key", "value"]


# ── Data (cached loaders) ─────────────────────────────────────────────────────
_loaders = None

def get_loaders():
    global _loaders
    if _loaders is not None:
        return _loaders
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
    val_size   = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_tf
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    _loaders = (train_loader, val_loader)
    return _loaders


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


def train_eval(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
    )
    model.to(DEVICE)
    for epoch in range(1, EPOCHS_HPO + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(imgs).logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds  = model(imgs).logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


def objective(trial: optuna.Trial):
    rank    = trial.suggest_categorical("rank",    [2, 4, 8])
    alpha   = trial.suggest_categorical("alpha",   [2, 4, 8])
    dropout = trial.suggest_float("dropout", 0.05, 0.2)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"trial_{trial.number}_r{rank}_a{alpha}_d{dropout:.2f}",
        config={"rank": rank, "alpha": alpha, "dropout": dropout},
        reinit=True,
    )

    train_loader, val_loader = get_loaders()
    model = build_model(rank, alpha, dropout)
    val_acc = train_eval(model, train_loader, val_loader)

    wandb.log({"val_acc": val_acc})
    wandb.finish()

    # Save best model candidate
    torch.save(model.state_dict(), f"optuna_trial_{trial.number}.pt")
    return val_acc


def main():
    study = optuna.create_study(
        direction="maximize",
        study_name="lora_vit_cifar100",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_trial
    print("\n=== Best Optuna Trial ===")
    print(f"  Val Accuracy : {best.value:.4f}")
    print(f"  Rank         : {best.params['rank']}")
    print(f"  Alpha        : {best.params['alpha']}")
    print(f"  Dropout      : {best.params['dropout']:.4f}")

    # ── Full retrain with best params and push to HuggingFace ────────────────
    from q1_push import retrain_and_push
    retrain_and_push(
        rank    = best.params["rank"],
        alpha   = best.params["alpha"],
        dropout = best.params["dropout"],
    )


if __name__ == "__main__":
    main()
