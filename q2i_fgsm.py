"""
Q2(i): FGSM Attack – From Scratch vs IBM ART
- ResNet18 trained from scratch on CIFAR-10 (target ≥ 72% clean acc)
- FGSM implemented manually
- FGSM via IBM ART
- Visual comparison + accuracy tables logged to WandB
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

# IBM ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 0.1
DATA_DIR    = "data"
WANDB_PROJECT = "FGSM-Attack"
EPSILONS    = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

# ── Data ──────────────────────────────────────────────────────────────────────
def get_cifar10():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, test_ds.classes, test_ds

# ── Model ─────────────────────────────────────────────────────────────────────
def build_resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ── Train ResNet18 from scratch ────────────────────────────────────────────────
def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            acc = evaluate(model, test_loader)
            print(f"Epoch {epoch:02d} | Clean Test Acc: {acc:.4f}")
            wandb.log({"epoch": epoch, "clean_test_acc": acc})

    torch.save(model.state_dict(), "resnet18_cifar10.pt")
    return model

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total

# ── FGSM from scratch ─────────────────────────────────────────────────────────
def fgsm_scratch(model, imgs, labels, eps):
    """Returns adversarial images using manual FGSM."""
    imgs   = imgs.clone().detach().to(DEVICE).requires_grad_(True)
    labels = labels.to(DEVICE)
    loss   = nn.CrossEntropyLoss()(model(imgs), labels)
    model.zero_grad()
    loss.backward()
    adv = imgs + eps * imgs.grad.sign()
    return adv.detach()

def evaluate_fgsm_scratch(model, loader, eps):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        adv = fgsm_scratch(model, imgs, labels, eps)
        with torch.no_grad():
            correct += (model(adv).argmax(1) == labels.to(DEVICE)).sum().item()
        total += labels.size(0)
    return correct / total

# ── FGSM via IBM ART ──────────────────────────────────────────────────────────
def build_art_classifier(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if DEVICE == "cuda" else "cpu",
    )
    return classifier

def evaluate_fgsm_art(art_clf, test_ds, eps, batch_size=256):
    """Run ART FGSM on the full test set."""
    x_test = np.array([np.array(test_ds[i][0]) for i in range(len(test_ds))], dtype=np.float32)
    y_test = np.array([test_ds[i][1] for i in range(len(test_ds))])

    # ART expects un-normalized (or normalized within clip range);
    # since our test_ds already applies Normalize, we pass directly.
    attack = FastGradientMethod(estimator=art_clf, eps=eps, batch_size=batch_size)
    x_adv  = attack.generate(x=x_test)

    preds = art_clf.predict(x_adv).argmax(axis=1)
    return (preds == y_test).mean()

# ── Visual comparison ─────────────────────────────────────────────────────────
def save_visual_comparison(model, art_clf, test_ds, classes, eps=0.05, n=10):
    """
    Shows 10 samples: Original | FGSM Scratch | FGSM ART
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)

    imgs_norm  = torch.stack([test_ds[i][0] for i in range(n)])
    labels_t   = torch.tensor([test_ds[i][1] for i in range(n)])

    adv_scratch = fgsm_scratch(model, imgs_norm, labels_t, eps).cpu()

    x_np  = imgs_norm.numpy()
    atk   = FastGradientMethod(estimator=art_clf, eps=eps, batch_size=n)
    x_adv = torch.tensor(atk.generate(x=x_np))

    def denorm(t):
        return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(3, n, figsize=(2*n, 6))
    for i in range(n):
        axes[0, i].imshow(denorm(imgs_norm[i]));     axes[0, i].axis("off")
        axes[1, i].imshow(denorm(adv_scratch[i]));   axes[1, i].axis("off")
        axes[2, i].imshow(denorm(x_adv[i]));         axes[2, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original",       fontsize=9)
            axes[1, i].set_ylabel("FGSM Scratch",   fontsize=9)
            axes[2, i].set_ylabel("FGSM ART",       fontsize=9)
        axes[0, i].set_title(classes[labels_t[i]], fontsize=7)

    plt.tight_layout()
    plt.savefig("fgsm_comparison.png", dpi=150)
    wandb.log({"FGSM_visual_comparison": wandb.Image("fgsm_comparison.png")})
    print("Saved fgsm_comparison.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wandb.init(project=WANDB_PROJECT, name="FGSM_scratch_vs_ART")

    train_loader, test_loader, classes, test_ds = get_cifar10()

    # 1. Train
    model = build_resnet18()
    model = train_model(model, train_loader, test_loader)
    clean_acc = evaluate(model, test_loader)
    print(f"\nClean Accuracy: {clean_acc:.4f}")
    assert clean_acc >= 0.72, f"Clean accuracy {clean_acc:.4f} below 72% target!"

    art_clf = build_art_classifier(model)

    # 2 & 3. FGSM for various epsilons
    print("\nEpsilon | Scratch Acc | ART Acc")
    print("-" * 38)
    eps_list, scratch_accs, art_accs = [], [], []
    for eps in EPSILONS:
        s_acc = evaluate_fgsm_scratch(model, test_loader, eps)
        a_acc = evaluate_fgsm_art(art_clf, test_ds, eps)
        print(f"  {eps:.3f}  |   {s_acc:.4f}   |  {a_acc:.4f}")
        wandb.log({"epsilon": eps, "fgsm_scratch_acc": s_acc, "fgsm_art_acc": a_acc})
        eps_list.append(eps); scratch_accs.append(s_acc); art_accs.append(a_acc)

    # Log accuracy table
    table = wandb.Table(
        columns=["Epsilon", "Clean Acc", "FGSM Scratch Acc", "FGSM ART Acc"],
        data=[[e, clean_acc, s, a] for e, s, a in zip(eps_list, scratch_accs, art_accs)]
    )
    wandb.log({"accuracy_table": table})

    # 4. Visual comparison at eps=0.05
    save_visual_comparison(model, art_clf, test_ds, classes, eps=0.05)

    wandb.finish()
    print("\nDone! Results logged to WandB.")


if __name__ == "__main__":
    main()
