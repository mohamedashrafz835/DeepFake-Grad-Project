import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =========================
# SE BLOCK
# =========================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(c // r, 4)),
            nn.ReLU(),
            nn.Linear(max(c // r, 4), c),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


# =========================
# ELA STREAM
# =========================
class ELAStream(nn.Module):
    def __init__(self, out=256):
        super().__init__()

        def blk(ci, co, s):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=s, padding=1, bias=False),
                nn.BatchNorm2d(co),
                nn.GELU(),
                SEBlock(co)
            )

        self.net = nn.Sequential(
            blk(3, 32, 1),
            blk(32, 64, 2),
            blk(64, 128, 2),
            blk(128, 256, 2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, out)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return F.gelu(self.fc(x))


# =========================
# MASK DECODER (OPTIONAL)
# =========================
class MaskDecoder(nn.Module):
    def __init__(self, in_c=1280):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(in_c, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.dec(x)


# =========================
# MAIN MODEL
# =========================
class ForgeryDetector(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)

        self.rgb_features = backbone.features
        self.rgb_pool = nn.AdaptiveAvgPool2d(1)

        self.ela_stream = ELAStream(256)
        self.mask_decoder = MaskDecoder(1280)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280 + 256, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),

            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),

            nn.Linear(128, 2)
        )

    def forward(self, rgb, ela):
        fm = self.rgb_features(rgb)

        rgb_feat = self.rgb_pool(fm).flatten(1)
        ela_feat = self.ela_stream(ela)

        x = torch.cat([rgb_feat, ela_feat], dim=1)

        return self.classifier(x)


# =========================
# MODEL LOADER (ROBUST)
# =========================
class ModelLoader:
    def __init__(self, model_path: str):
        self.device = "cpu"

        self.model = ForgeryDetector().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            else:
                state = checkpoint
        else:
            raise ValueError("Invalid checkpoint format")

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict(self, rgb, ela):
        with torch.no_grad():
            logits = self.model(rgb, ela)
            probs = torch.softmax(logits, dim=1)

        return probs