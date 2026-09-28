import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from model import get_model, DEVICE
from utils import make_file_label_list, FoodDataset

def parse_args():
    p = argparse.ArgumentParser(description="Training Model Klasifikasi Citra Makanan")
    p.add_argument("--train_data", type=str, default="data/train", help="Folder data pelatihan")
    p.add_argument("--val_data", type=str, default="data/val", help="Folder data validasi (opsional)")
    p.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"], help="Arsitektur model")
    p.add_argument("--bs", type=int, default=16, help="Ukuran batch")
    p.add_argument("--epochs", type=int, default=15, help="Jumlah epoch")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--out", type=str, default="checkpoints", help="Folder output checkpoint & hasil")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Penyiapan Dataset
    if os.path.exists(args.val_data) and len(os.listdir(args.val_data)) > 0:
        print(f"[*] Menggunakan folder training terpisah: '{args.train_data}' dan validasi: '{args.val_data}'")
        f_train, l_train, classes_train = make_file_label_list(args.train_data)
        f_val, l_val, classes_val = make_file_label_list(args.val_data)
        classes = classes_train
    else:
        print(f"[*] Membagi dataset dari '{args.train_data}' (80% train, 20% val)...")
        files, labels, classes = make_file_label_list(args.train_data)
        f_train, f_val, l_train, l_val = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=42)

    print(f"[*] Jumlah kelas: {len(classes)} -> {classes}")
    print(f"[*] Total data latih: {len(f_train)} | Total data validasi: {len(f_val)}")

    # 2. Data Augmentasi & Transformasi Citra
    input_size = 224
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FoodDataset(f_train, l_train, train_transform)
    val_ds = FoodDataset(f_val, l_val, val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    # 3. Model, Loss Function, Optimizer & Scheduler
    model = get_model(len(classes), args.arch, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }

    best_val_acc = 0.0

    print(f"\n[*] Memulai Pelatihan ({args.arch}) selama {args.epochs} Epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # --- Fase Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labs in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labs).sum().item()
            train_total += labs.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --- Fase Validasi ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, labs)
                val_loss += loss.item() * imgs.size(0)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labs).sum().item()
                val_total += labs.size(0)

                y_true.extend(labs.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        scheduler.step(epoch_val_loss)

        # Hitung Precision, Recall, F1
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_precision"].append(float(p_macro))
        history["val_recall"].append(float(r_macro))
        history["val_f1"].append(float(f1_macro))

        print(f"Epoch {epoch:02d} -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | F1: {f1_macro:.4f}")

        # Simpan checkpoint epoch
        ckpt = {
            "epoch": epoch,
            "arch": args.arch,
            "classes": classes,
            "model_state": model.state_dict(),
            "val_acc": epoch_val_acc,
            "val_loss": epoch_val_loss
        }
        torch.save(ckpt, os.path.join(args.out, f"ckpt_epoch{epoch}.pth"))

        # Simpan Model Terbaik
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(ckpt, os.path.join(args.out, "best_model.pth"))
            print(f"[*] Rekor Akurasi Baru ({best_val_acc*100:.2f}%) tersimpan di {args.out}/best_model.pth")

            # Simpan Confusion Matrix Model Terbaik
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title(f"Confusion Matrix Terbaik - Epoch {epoch} (Akurasi: {best_val_acc*100:.2f}%)")
            plt.tight_layout()
            fig.savefig(os.path.join(args.out, "best_confusion_matrix.png"))
            plt.close(fig)

            # Simpan Classification Report teks
            report_str = classification_report(y_true, y_pred, target_names=classes, digits=4)
            with open(os.path.join(args.out, "best_classification_report.txt"), "w") as f:
                f.write(report_str)

    # 4. Simpan Riwayat Metrik & Kurva Pembelajaran (Learning Curves)
    with open(os.path.join(args.out, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, args.epochs + 1)

    # Loss Curve
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker='s')
    ax1.set_title(f"Kurva Loss ({args.arch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy Curve
    ax2.plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Accuracy", marker='o')
    ax2.plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Accuracy", marker='s')
    ax2.set_title(f"Kurva Akurasi ({args.arch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Akurasi (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "learning_curves.png"))
    plt.close(fig)

    print(f"\n[V] Pelatihan selesai. Model terbaik: {best_val_acc*100:.2f}%")
    print(f"[V] Grafik kurva pelatihan disimpan di '{args.out}/learning_curves.png'")

if __name__ == "__main__":
    main()
