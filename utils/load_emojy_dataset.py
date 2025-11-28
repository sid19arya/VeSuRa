from datasets import load_dataset
import torch
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random
from typing import List, Tuple, Optional
import io
from svg_load import canonicalize_svg
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

# Augmentation functions
def get_augmentations():
    return [
        transforms.Compose([]),  # Identity (no augmentation)
        transforms.Compose([transforms.RandomRotation(degrees=90)]),
        transforms.Compose([transforms.GaussianBlur(kernel_size=11, sigma=(2.0, 5.0))]),
        transforms.Compose([transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2)]),
        transforms.Compose([transforms.RandomAffine(degrees=60, translate=(0.2, 0.2), scale=(0.7, 1.3), shear=30)]),
        transforms.Compose([transforms.RandomPerspective(distortion_scale=0.7, p=1.0)]),
        transforms.Compose([transforms.RandomInvert(p=1.0)]),
    ]

# Utility to load EmojySVG dataset from a directory structure with PNG and SVG files
def load_emojysvg_dataset(data_dir, max_curves=128, max_paths_per_curve=16, viewbox_size=128):
    """
    Loads a dataset from a directory with paired .png and .svg files.
    Returns a list of dicts with keys: 'ground_truth_image', 'image', 'svg_params', 'svg_mask', 'svg_path', 'img_path'
    """
    items = []
    augmentations = get_augmentations()
    for fname in os.listdir(data_dir):
        if fname.endswith('.png'):
            base = fname[:-4]
            img_path = os.path.join(data_dir, base + '.png')
            svg_path = os.path.join(data_dir, base + '.svg')
            if not os.path.exists(svg_path):
                continue
            # Load image
            img = Image.open(img_path).convert('RGB').resize((viewbox_size, viewbox_size))
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3], RGB
            # Load SVG and canonicalize
            with open(svg_path, 'r') as f:
                try:
                    bez, mask = canonicalize_svg(f, max_curves, max_paths_per_curve, viewbox_size)
                except Exception as e:
                    print(f'Error parsing {svg_path}: {e}')
                    bez, mask = None, None
            # Apply augmentations
            for aug in augmentations:
                aug_img = aug(img)
                aug_img_np = np.array(aug_img)  # Ensure still RGB, shape [H, W, 3]
                aug_img_tensor = torch.from_numpy(aug_img_np).float() / 255.0
                items.append({
                    'ground_truth_image': img_tensor,  # always original
                    'image': aug_img_tensor,
                    'svg_params': bez,
                    'svg_mask': mask,
                    'svg_path': svg_path,
                    'img_path': img_path
                })
    return items


class EmojySVGDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            'ground_truth_image': item['ground_truth_image'].permute(2, 0, 1),
            'image': item['image'].permute(2, 0, 1),
            'svg_params': item['svg_params'],
            'svg_mask': item['svg_mask']
        }

def get_emojysvg_dataloaders(data_dir, batch_size=4, train_split=0.8, max_curves=128, max_paths_per_curve=16, viewbox_size=128):
    emojy_dataset = load_emojysvg_dataset(data_dir, max_curves=max_curves, max_paths_per_curve=max_paths_per_curve, viewbox_size=viewbox_size)
    emojy_torch_dataset = EmojySVGDataset(emojy_dataset)

    train_size = int(train_split * len(emojy_torch_dataset))
    test_size = len(emojy_torch_dataset) - train_size

    train_set, test_set = random_split(emojy_torch_dataset, [train_size, test_size])

    emojy_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    emojy_test_loader   = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return emojy_torch_dataset, train_set, test_set, emojy_train_loader, emojy_test_loader
