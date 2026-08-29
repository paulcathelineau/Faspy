"""Semantic route: a compact U-Net over three classes.

This route segments the section footprint extremely well and is what defines the
leaflet area used as the denominator of every ratio. It is not used to quantify
bundles: pixel-wise labelling merges touching bundles and paints large spurious
patches in the mesophyll, which no area filter can undo. Bundles are handled by
the instance route instead.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import (
    BASE_CHANNELS,
    BATCH_SIZE,
    BUNDLE_CLASS,
    CLASS_WEIGHTS,
    IGNORE_INDEX,
    INFER_STRIDE,
    LEARNING_RATE,
    MIN_BUNDLE_AREA,
    N_CLASSES,
    PATCH,
    PATCHES_PER_IMAGE,
    POSTPROCESS_BUNDLE,
    SEED,
)
from .datasets import PatchDataset


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Four-level encoder/decoder with skip connections, trained from scratch.

    An ImageNet-pretrained encoder was evaluated and performed worse here: the
    domain gap to stained microscopy outweighs the transfer, and the pretrained
    weights degrade at any learning rate high enough to adapt them.
    """

    def __init__(self, channels_in=3, n_classes=N_CLASSES, base=BASE_CHANNELS):
        super().__init__()
        self.down1 = DoubleConv(channels_in, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.upsample4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.up4 = DoubleConv(base * 16, base * 8)
        self.upsample3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.up3 = DoubleConv(base * 8, base * 4)
        self.upsample2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up2 = DoubleConv(base * 4, base * 2)
        self.upsample1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.up1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool(skip1))
        skip3 = self.down3(self.pool(skip2))
        skip4 = self.down4(self.pool(skip3))
        x = self.bottleneck(self.pool(skip4))

        x = self.up4(torch.cat([self.upsample4(x), skip4], dim=1))
        x = self.up3(torch.cat([self.upsample3(x), skip3], dim=1))
        x = self.up2(torch.cat([self.upsample2(x), skip2], dim=1))
        x = self.up1(torch.cat([self.upsample1(x), skip1], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, ignore_index=IGNORE_INDEX, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        probabilities = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe = target.clone()
        safe[~valid] = 0

        one_hot = F.one_hot(safe, self.n_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1).float()
        probabilities = probabilities * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        union = probabilities.sum(dims) + one_hot.sum(dims)
        return 1 - ((2 * intersection + self.eps) / (union + self.eps)).mean()


class ComboLoss(nn.Module):
    """Weighted cross-entropy plus Dice.

    Cross-entropy alone under-segments the bundle class, which covers only a few
    per cent of the pixels; Dice alone is unstable early in training.
    """

    def __init__(self):
        super().__init__()
        weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()

    def forward(self, logits, target):
        return self.cross_entropy(logits, target) + self.dice(logits, target)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(rows, device, epochs, batch_size=BATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE,
          verbose=True):
    """Train a U-Net on the given manifest rows and return it."""
    torch.manual_seed(SEED)
    loader = DataLoader(
        PatchDataset(rows, patches_per_image=patches_per_image, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = UNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ComboLoss().to(device)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            optimiser.zero_grad()
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(image), mask)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item()
        if verbose:
            print(f"    epoch {epoch}/{epochs}  loss {running / len(loader):.3f}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, rgb: np.ndarray, device: str) -> np.ndarray:
    """Full-section prediction by averaging softmax over a sliding window."""
    height, width = rgb.shape[:2]
    patch = PATCH
    padded = cv2.copyMakeBorder(
        rgb, 0, max(0, patch - height), 0, max(0, patch - width), cv2.BORDER_CONSTANT
    )
    padded_height, padded_width = padded.shape[:2]

    accumulated = np.zeros((N_CLASSES, padded_height, padded_width), dtype=np.float32)
    counts = np.zeros((padded_height, padded_width), dtype=np.float32)

    rows = sorted(set(list(range(0, max(1, padded_height - patch + 1), INFER_STRIDE))
                      + [padded_height - patch]))
    columns = sorted(set(list(range(0, max(1, padded_width - patch + 1), INFER_STRIDE))
                         + [padded_width - patch]))

    for y in rows:
        for x in columns:
            window = padded[y:y + patch, x:x + patch]
            tensor = torch.from_numpy(window.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            output = torch.softmax(model(tensor.to(device)), dim=1)[0].cpu().numpy()
            accumulated[:, y:y + patch, x:x + patch] += output
            counts[y:y + patch, x:x + patch] += 1

    accumulated /= np.maximum(counts, 1e-6)
    return accumulated.argmax(0).astype(np.uint8)[:height, :width]


def postprocess(label: np.ndarray, min_area: int = MIN_BUNDLE_AREA) -> np.ndarray:
    """Close small gaps inside bundles and drop components below ``min_area``.

    Sliding-window prediction leaves thin seams across large bundles and a
    scatter of small false patches. Closing repairs the first; the area filter,
    defined at full resolution in the configuration, removes the second.
    """
    if not POSTPROCESS_BUNDLE:
        return label

    bundle = (label == BUNDLE_CLASS).astype(np.uint8)
    if not bundle.any():
        return label

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(bundle, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros_like(closed, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == index

    out = label.copy()
    out[out == BUNDLE_CLASS] = 1          # demote every bundle pixel to leaflet
    out[keep & (out == 1)] = BUNDLE_CLASS  # then reinstate the surviving ones
    return out


def colourise(label: np.ndarray) -> np.ndarray:
    """Render a label map as a BGR overlay."""
    from .config import CLASS_COLOURS

    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for value, colour in CLASS_COLOURS.items():
        out[label == value] = colour
    return out
