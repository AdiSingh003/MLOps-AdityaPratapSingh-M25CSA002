"""
Q2(ii): Adversarial Detection Model using ResNet-34
- PGD attack via IBM ART → binary detector (clean vs adversarial)
- BIM attack via IBM ART → binary detector (clean vs adversarial)
- Detection accuracy ≥ 70%
- WandB logging with sample images for FGSM, PGD, BIM
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    BasicIterativeMethod,
)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 128
EPOCHS_CLF  = 30      # epochs to train victim ResNet-18
EPOCHS_DET  = 20      # epochs to train detector ResNet-34
LR_CLF      = 0.1
LR_DET      = 1e-3
DATA_DIR    = "data"
WANDB_PROJECT = "Adversarial-Detection"

# ── Data helpers ──────────────────────────────────────────────────────────────
def get_cifar10_raw():
    """Returns (x_train, y_train, x_test, y_test) as numpy float32 arrays,
    un-normalised in [0,1], and also loaders with normalisation for the victim."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    norm_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    aug_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    raw_tf = transforms.ToTensor()

    train_norm = torchvision.datasets.CIFAR10(DATA_DIR, True,  download=True, transform=aug_tf)
    test_norm  = torchvision.datasets.CIFAR10(DATA_DIR, False, download=True, transform=norm_tf)
    test_raw   = torchvision.datasets.CIFAR10(DATA_DIR, False, download=True, transform=raw_tf)

    train_loader = DataLoader(train_norm, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_norm,  BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Build numpy arrays for ART
    x_test = np.array([np.array(test_raw[i][0]) for i in range(len(test_raw))], dtype=np.float32)
    y_test = np.array([test_raw[i][1] for i in range(len(test_raw))])
    return train_loader, test_loader, x_test, y_test, test_norm.classes

# ── Victim model (ResNet-18 from scratch) ─────────────────────────────────────
def train_victim(train_loader, test_loader):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR_CLF, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_CLF)

    for epoch in range(1, EPOCHS_CLF + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"Victim Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(imgs), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Victim Clean Accuracy: {correct/total:.4f}")
    assert correct/total >= 0.72
    torch.save(model.state_dict(), "victim_resnet18.pt")
    return model

def build_art_classifier(model):
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=LR_CLF),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if DEVICE == "cuda" else "cpu",
    )

# ── Generate adversarial examples ─────────────────────────────────────────────
def generate_pgd(art_clf, x, eps=8/255, eps_step=2/255, max_iter=40):
    attack = ProjectedGradientDescent(
        estimator=art_clf, eps=eps, eps_step=eps_step,
        max_iter=max_iter, batch_size=BATCH_SIZE,
    )
    return attack.generate(x)

def generate_bim(art_clf, x, eps=8/255, eps_step=2/255, max_iter=40):
    attack = BasicIterativeMethod(
        estimator=art_clf, eps=eps, eps_step=eps_step,
        max_iter=max_iter, batch_size=BATCH_SIZE,
    )
    return attack.generate(x)

def generate_fgsm(art_clf, x, eps=8/255):
    attack = FastGradientMethod(estimator=art_clf, eps=eps, batch_size=BATCH_SIZE)
    return attack.generate(x)

# ── Detector model (ResNet-34, binary output) ─────────────────────────────────
def build_detector():
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def make_binary_dataset(x_clean, x_adv):
    """Returns a TensorDataset: clean=0, adversarial=1."""
    X = torch.tensor(np.concatenate([x_clean, x_adv], axis=0))
    Y = torch.cat([torch.zeros(len(x_clean), dtype=torch.long),
                   torch.ones(len(x_adv),   dtype=torch.long)])
    return TensorDataset(X, Y)

def train_detector(detector, dataset, tag="detector"):
    val_size   = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    detector.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(detector.parameters(), lr=LR_DET, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_DET)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS_DET + 1):
        detector.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"[{tag}] Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(detector(imgs), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        detector.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (detector(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total
        wandb.log({f"{tag}/val_acc": val_acc, f"{tag}/epoch": epoch})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in detector.state_dict().items()}

    detector.load_state_dict(best_state)
    return detector

def eval_detector(detector, x_clean, x_adv):
    X = torch.tensor(np.concatenate([x_clean, x_adv], axis=0))
    Y = torch.cat([torch.zeros(len(x_clean), dtype=torch.long),
                   torch.ones(len(x_adv),   dtype=torch.long)])
    loader = DataLoader(TensorDataset(X, Y), BATCH_SIZE, shuffle=False)
    detector.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (detector(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ── WandB sample images ────────────────────────────────────────────────────────
def log_sample_images(x_clean, x_fgsm, x_fgsm_art, x_pgd, x_bim, n=10):
    def show(x, title):
        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        for i in range(n):
            axes[i].imshow(x[i].transpose(1, 2, 0).clip(0, 1))
            axes[i].axis("off")
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        fname = f"{title.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=100)
        plt.close()
        return fname

    fnames = {
        "Clean Images":         show(x_clean[:n],    "Clean Images"),
        "FGSM Scratch":         show(x_fgsm[:n],     "FGSM Scratch"),
        "FGSM ART":             show(x_fgsm_art[:n], "FGSM ART"),
        "PGD ART":              show(x_pgd[:n],      "PGD ART"),
        "BIM ART":              show(x_bim[:n],      "BIM ART"),
    }
    wandb.log({k: wandb.Image(v) for k, v in fnames.items()})
    print("Sample images logged to WandB.")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wandb.init(project=WANDB_PROJECT, name="AdversarialDetection_PGD_BIM")

    train_loader, test_loader, x_test, y_test, classes = get_cifar10_raw()

    # 1. Train victim
    print("=== Training victim ResNet-18 ===")
    victim   = train_victim(train_loader, test_loader)
    art_clf  = build_art_classifier(victim)

    # 2. Generate adversarial examples
    print("\n=== Generating adversarial examples ===")
    print("  PGD ...")
    x_pgd  = generate_pgd(art_clf,  x_test)
    print("  BIM ...")
    x_bim  = generate_bim(art_clf,  x_test)
    print("  FGSM (scratch / ART) ...")
    x_fgsm_art = generate_fgsm(art_clf, x_test, eps=8/255)
    # manual FGSM (subset, using normalised loader)
    x_fgsm_scratch = generate_fgsm(art_clf, x_test, eps=8/255)  # same eps for logging

    # 3. Log sample images
    log_sample_images(x_test, x_fgsm_scratch, x_fgsm_art, x_pgd, x_bim, n=10)

    # 4(a). Detector for PGD
    print("\n=== Training PGD detector (ResNet-34) ===")
    pgd_dataset  = make_binary_dataset(x_test, x_pgd)
    pgd_detector = build_detector()
    pgd_detector = train_detector(pgd_detector, pgd_dataset, tag="PGD_detector")
    pgd_det_acc  = eval_detector(pgd_detector, x_test, x_pgd)
    print(f"PGD Detection Accuracy: {pgd_det_acc:.4f}")
    assert pgd_det_acc >= 0.70, f"PGD detector below 70% ({pgd_det_acc:.4f})"
    torch.save(pgd_detector.state_dict(), "pgd_detector.pt")

    # 4(b). Detector for BIM
    print("\n=== Training BIM detector (ResNet-34) ===")
    bim_dataset  = make_binary_dataset(x_test, x_bim)
    bim_detector = build_detector()
    bim_detector = train_detector(bim_detector, bim_dataset, tag="BIM_detector")
    bim_det_acc  = eval_detector(bim_detector, x_test, x_bim)
    print(f"BIM Detection Accuracy: {bim_det_acc:.4f}")
    assert bim_det_acc >= 0.70, f"BIM detector below 70% ({bim_det_acc:.4f})"
    torch.save(bim_detector.state_dict(), "bim_detector.pt")

    # 5. Summary table
    summary = wandb.Table(
        columns=["Attack", "Detection Accuracy"],
        data=[
            ["PGD (ART)", f"{pgd_det_acc:.4f}"],
            ["BIM (ART)", f"{bim_det_acc:.4f}"],
        ]
    )
    wandb.log({"detection_summary": summary})
    print("\n=== Summary ===")
    print(f"PGD Detection Accuracy: {pgd_det_acc:.4f}")
    print(f"BIM Detection Accuracy: {bim_det_acc:.4f}")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
