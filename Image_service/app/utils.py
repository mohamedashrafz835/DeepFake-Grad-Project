from PIL import Image, ImageChops, ImageEnhance
import torch
from torchvision import transforms


# =========================
# FIXED ELA (MATCH TRAINING)
# =========================
def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15):
    tmp = "/tmp/tmp_ela.jpg"

    image.save(tmp, "JPEG", quality=quality)
    comp = Image.open(tmp).convert("RGB")

    ela = ImageChops.difference(image, comp)

    extrema = ela.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1

    # 🔥 MUST MATCH TRAINING
    ela = ImageEnhance.Brightness(ela).enhance(scale * 255.0 / max_diff)

    return ela


# =========================
# RGB TRANSFORM (FIXED)
# =========================
rgb_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# ELA TRANSFORM (FIXED)
# =========================
ela_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


# =========================
# PREPROCESS (CONSISTENT)
# =========================
def preprocess(img: Image.Image):
    img = img.convert("RGB")

    ela = compute_ela(img)

    rgb_tensor = rgb_tf(img).unsqueeze(0)
    ela_tensor = ela_tf(ela).unsqueeze(0)

    return rgb_tensor, ela_tensor