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

# Utility to load EmojySVG dataset from a directory structure with PNG and SVG files
def load_emojysvg_dataset(data_dir, max_curves=128, max_paths_per_curve=16, viewbox_size=128):
    """
    Loads a dataset from a directory with paired .png and .svg files.
    Returns a list of dicts with keys: 'sketch_image', 'svg_params', 'svg_mask', 'svg_path', 'img_path'
    """
    items = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.png'):
            base = fname[:-4]
            img_path = os.path.join(data_dir, base + '.png')
            svg_path = os.path.join(data_dir, base + '.svg')
            if not os.path.exists(svg_path):
                continue
            # Load image
            img = Image.open(img_path).convert('RGB').resize((viewbox_size, viewbox_size))
            img_tensor = torch.tensor(np.array(img)/255., dtype=torch.float32)
            # Load SVG and canonicalize
            with open(svg_path, 'r') as f:
              try:
                  bez, mask = canonicalize_svg(f, max_curves, max_paths_per_curve, viewbox_size)
              except Exception as e:
                  print(f'Error parsing {svg_path}: {e}')
                  bez, mask = None, None
            items.append({
                'sketch_image': img_tensor,
                'svg_params': bez,
                'svg_mask': mask,
                'svg_path': svg_path,
                'img_path': img_path
            })
    return items