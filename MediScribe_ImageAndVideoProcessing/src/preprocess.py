#!/usr/bin/env python3
"""Image preprocessing utilities: resize, CLAHE, denoise helpers."""
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def resize_and_pad(img, short_side=384):
    # Resize keeping aspect, pad to square short_side x short_side or to ratio 1:1
    w, h = img.size
    scale = short_side / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # center pad
    pad_w = max(0, short_side - new_w)
    pad_h = max(0, short_side - new_h)
    left = pad_w // 2
    top = pad_h // 2
    result = Image.new('RGB', (max(short_side, new_w), max(short_side, new_h)), (255,255,255))
    result.paste(img, (left, top))
    return result

def clahe_pil(img):
    # Convert to LAB and apply CLAHE on L channel
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final)
