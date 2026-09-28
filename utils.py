import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def make_file_label_list(root_dir):
    """
    Membaca struktur direktori berbasis kelas:
    root_dir/
      ├── Bread/
      ├── Noodles-Pasta/
      ├── Seafood/
      └── nasi-goreng/
    Mengabaikan file tersembunyi seperti .DS_Store.
    """
    files = []
    labels = []
    if not os.path.exists(root_dir):
        return files, labels, []

    classes = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for c in classes:
        cdir = os.path.join(root_dir, c)
        for f in os.listdir(cdir):
            if f.lower().endswith(valid_exts) and not f.startswith("."):
                files.append(os.path.join(cdir, f))
                labels.append(class_to_idx[c])

    return files, labels, classes

def predict_image(model, pil_img, transform, classes):
    """
    Melakukan inferensi klasifikasi citra makanan:
    1. Preprocessing citra dengan transform
    2. Forward pass ke model CNN
    3. Normalisasi logit dengan Softmax untuk mendapatkan probabilitas kepercayaan (Confidence Score).
    """
    model.eval()
    x = transform(pil_img).unsqueeze(0)
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return classes[idx], float(probs[idx]), probs, logits.cpu().numpy()[0]
