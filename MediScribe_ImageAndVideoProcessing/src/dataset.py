#!/usr/bin/env python3
"""
Custom PyTorch Dataset for TrOCR training with prescription images.
Optimized for RTX 4060 hardware constraints.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PrescriptionDataset(Dataset):
    """Custom dataset for prescription OCR training with augmentations."""
    
    def __init__(
        self, 
        jsonl_path: str, 
        processor, 
        max_length: int = 256,
        image_size: int = 384,
        augment: bool = True,
        debug: bool = False
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.augment = augment
        self.debug = debug
        
        # Load data from JSONL
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if os.path.exists(item['image_path']) and item['text'].strip():
                    self.data.append(item)
        
        print(f"Loaded {len(self.data)} valid examples from {jsonl_path}")
        
        # Setup augmentations for training
        if augment:
            self.transform = A.Compose([
                # Geometric augmentations (keep minimal for prescription OCR)
                A.Rotate(limit=3, p=0.3),  # Very small rotation
                A.Perspective(scale=(0.02, 0.05), p=0.2),  # Slight perspective
                
                # Color/lighting augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, 
                    contrast_limit=0.15, 
                    p=0.4
                ),
                A.GaussNoise(var_limit=(5, 15), p=0.2),
                A.GaussianBlur(blur_limit=(1, 3), p=0.1),
                
                # Medical prescription specific augmentations
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.1),
                
                # Normalization (will be handled by processor later)
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply prescription-specific preprocessing."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Apply CLAHE for better contrast (medical documents often have poor contrast)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_img = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2RGB)
        
        # Light denoising (preserve text details)
        enhanced_img = cv2.medianBlur(enhanced_img, 3)
        
        return Image.fromarray(enhanced_img)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load and preprocess image
            image = Image.open(item['image_path'])
            image = self.preprocess_image(image)
            
            # Apply augmentations if training
            if self.transform:
                # Convert to numpy for albumentations
                img_array = np.array(image)
                augmented = self.transform(image=img_array)
                image = Image.fromarray(augmented['image'])
            
            # Process with TrOCR processor
            encoding = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"height": self.image_size, "width": self.image_size}
            )
            
            # Process text labels
            labels = self.processor.tokenizer(
                item['text'],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0),
                "confidence": item.get('conf', 1.0)  # For weighted training
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error processing {item['image_path']}: {e}")
            
            # Return a dummy example to avoid training interruption
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            encoding = self.processor(images=dummy_image, return_tensors="pt")
            labels = self.processor.tokenizer(
                "ERROR", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0),
                "confidence": 0.0
            }


class PrescriptionDataCollator:
    """Custom data collator for prescription OCR."""
    
    def __init__(self, processor, pad_token_id: int = None):
        self.processor = processor
        self.pad_token_id = pad_token_id or processor.tokenizer.pad_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract components
        pixel_values = [f["pixel_values"] for f in features]
        labels = [f["labels"] for f in features]
        confidences = [f["confidence"] for f in features]
        
        # Stack pixel values
        pixel_values = torch.stack(pixel_values)
        
        # Handle labels (replace pad tokens with -100 for loss calculation)
        labels = torch.stack(labels)
        labels[labels == self.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "confidences": torch.tensor(confidences, dtype=torch.float32)
        }


def create_datasets(train_jsonl: str, val_jsonl: str, processor, **kwargs):
    """Create train and validation datasets."""
    
    train_dataset = PrescriptionDataset(
        train_jsonl, 
        processor, 
        augment=True,
        **kwargs
    )
    
    val_dataset = PrescriptionDataset(
        val_jsonl, 
        processor, 
        augment=False,  # No augmentation for validation
        **kwargs
    )
    
    return train_dataset, val_dataset


def split_dataset(jsonl_path: str, train_ratio: float = 0.8, seed: int = 42):
    """Split a single JSONL file into train/val splits."""
    import random
    random.seed(seed)
    
    # Load all data
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Shuffle and split
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Save splits
    train_path = jsonl_path.replace('.jsonl', '_train.jsonl')
    val_path = jsonl_path.replace('.jsonl', '_val.jsonl')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Split {len(data)} examples into {len(train_data)} train, {len(val_data)} val")
    return train_path, val_path