"""Grad-CAM für das ResNet-18 aus resnet_evaluation.ipynb.

Grad-CAM gewichtet die Aktivierungen einer Faltungsschicht mit den
Gradienten einer Zielklasse und zeigt so, welche Bildregionen die
Vorhersage treiben. Wird vom Notebook (Abschnitt 14) und von
gradcam_tool.py gemeinsam verwendet.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_resnet18_checkpoint(path: Path, device: torch.device) -> nn.Module:
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)


def load_patch(row, patch_size: int = 128) -> np.ndarray:
    """Baut den Bauteil-Crop aus dem Originalbild nach (wie im Notebook)."""
    arr = np.asarray(Image.open(row["image_path"]).convert("L"), dtype=np.uint8)
    height, width = arr.shape
    size = int(row["crop_side"])
    center_y, center_x = int(row["center_y"]), int(row["center_x"])
    half_before = size // 2
    half_after = size - half_before

    y1 = max(0, center_y - half_before)
    y2 = min(height, center_y + half_after)
    x1 = max(0, center_x - half_before)
    x2 = min(width, center_x + half_after)
    crop = arr[y1:y2, x1:x2]

    pad_top = max(0, half_before - center_y)
    pad_bottom = max(0, center_y + half_after - height)
    pad_left = max(0, half_before - center_x)
    pad_right = max(0, center_x + half_after - width)
    if any(v > 0 for v in [pad_top, pad_bottom, pad_left, pad_right]):
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    return np.asarray(Image.fromarray(crop).resize((patch_size, patch_size), Image.Resampling.BILINEAR), dtype=np.uint8)


def patch_to_tensor(patch: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(patch.copy()).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0).to(device)


class GradCAM:
    """Klassen-Aktivierungskarte über die Gradienten der Ziel-Schicht.

    Die Karte wird je Bild auf [0, 1] normiert; sie zeigt also die relative
    Verteilung der Aufmerksamkeit, nicht deren absolute Stärke.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = target_layer if target_layer is not None else model.layer4
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, args, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> tuple[np.ndarray, float]:
        logits = self.model(x)
        self.model.zero_grad()
        logits[:, class_idx].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-6)

        prob_anomaly = torch.softmax(logits.detach(), dim=1)[0, 1].item()
        return cam, prob_anomaly
