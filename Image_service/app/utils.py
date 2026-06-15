from PIL import Image, ImageChops, ImageEnhance
import torch
from torchvision import transforms


def compute_ela(image):
    tmp = "/tmp/tmp.jpg"
    image.save(tmp, "JPEG", quality=90)

    comp = Image.open(tmp)
    ela = ImageChops.difference(image, comp)

    extrema = ela.getextrema()
    max_diff = max([e[1] for e in extrema]) or 1

    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
    return ela


rgb_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

ela_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def preprocess(img):
    img = img.convert("RGB")

    ela = compute_ela(img)

    rgb = rgb_tf(img).unsqueeze(0)
    ela = ela_tf(ela).unsqueeze(0)

    return rgb, ela
