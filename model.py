import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(num_classes, arch="resnet50", pretrained=True):
    """
    Menginisialisasi arsitektur deep learning (ResNet50 atau EfficientNet-B0)
    dengan Transfer Learning (Pretrained weights di ImageNet)
    dan menyesuaikan lapisan klasifikasi akhir (Fully Connected / Classifier Layer).
    """
    if arch == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Arsitektur '{arch}' tidak didukung. Pilih 'resnet50' atau 'efficientnet_b0'.")

    return model.to(DEVICE)

def get_inference_transforms(input_size=224):
    """Transformasi preprocessing standar untuk inferensi (Resize & Normalisasi ImageNet)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def load_checkpoint(path, num_classes, arch="resnet50"):
    """Memuat bobot model dari file checkpoint (.pth)."""
    model = get_model(num_classes, arch=arch, pretrained=False)
    ckpt = torch.load(path, map_location=DEVICE)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model
