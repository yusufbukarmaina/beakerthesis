"""
Data utilities for loading and processing the beaker dataset.
Uses LAZY loading - images are NOT loaded into RAM upfront.

QWEN2-VL label strategy:
  Qwen2-VL expects labels to be the SAME length as input_ids.
  Positions belonging to the prompt (input) are masked with -100.
  Only the answer tokens contribute to the loss.
"""

import re
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split

from config import (
    DATASET_NAME, DATASET_SPLIT_SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    TEST_DATA_DIR, MAX_LENGTH,
    FLORENCE_PROMPT_TEMPLATE, QWEN_PROMPT_TEMPLATE
)

# ── Constants ────────────────────────────────────────────────────────────────

VALID_VIEWS = {'l', 'r', 'a', 'b', 'f'}
VALID_BGS   = {'c', 'u'}

BG_MAP = {
    'c': 'controlled', 'u': 'uncontrolled', 'unknown': 'unknown'
}
VIEW_MAP = {
    'l': 'left', 'r': 'right', 'a': 'above',
    'b': 'below', 'f': 'front', 'unknown': 'unknown'
}


# ── Filename Parser ──────────────────────────────────────────────────────────

def parse_filename(filename: str) -> Optional[Dict]:
    """Robust parser handling all observed filename variations."""
    name = filename.lower().replace('.jpg','').replace('.jpeg','').replace('.png','')
    parts = name.split('_')
    if len(parts) < 3:
        return None
    try:
        image_id  = parts[0]
        cap_match = re.match(r'^(\d+)ml$', parts[1])
        if not cap_match:
            return None
        beaker_capacity = int(cap_match.group(1))

        vol_match = re.match(r'^v?(\d+\.?\d*)ml$', parts[2])
        if not vol_match:
            return None
        liquid_volume = float(vol_match.group(1))

        background = 'unknown'
        viewpoint  = 'unknown'
        remaining  = parts[3:]

        def clean_view(t):
            t = t.rstrip('0123456789')
            if len(t) > 1 and len(set(t)) == 1:
                t = t[0]
            return t[0] if t else ''

        if len(remaining) == 1:
            tok = clean_view(remaining[0])
            if tok in VALID_VIEWS: viewpoint  = tok
            elif tok in VALID_BGS: background = tok
        elif len(remaining) == 2:
            if remaining[0] in VALID_BGS: background = remaining[0]
            v = clean_view(remaining[1])
            if v in VALID_VIEWS: viewpoint = v
        elif len(remaining) > 2:
            if remaining[0] in VALID_BGS: background = remaining[0]
            v = clean_view(remaining[-1])
            if v in VALID_VIEWS: viewpoint = v

        return {
            'id':              image_id,
            'beaker_capacity': beaker_capacity,
            'liquid_volume':   liquid_volume,
            'background':      BG_MAP.get(background, background),
            'viewpoint':       VIEW_MAP.get(viewpoint, viewpoint),
        }
    except Exception:
        return None


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_and_split_dataset(cache_dir: Optional[str] = None) -> dict:
    """
    Load dataset from HuggingFace — metadata only, no images in RAM.
    Returns dict: {'train': {'meta': [...], 'raw_split': ...}, ...}
    """
    print(f"Loading dataset: {DATASET_NAME}")
    print("(Using lazy image loading - metadata only pass)")

    raw = load_dataset(DATASET_NAME, cache_dir=cache_dir, keep_in_memory=False)

    splits_out = {}
    total_skip = 0

    for split_name in ['train', 'validation', 'test']:
        if split_name not in raw:
            print(f"  Warning: split '{split_name}' not found, skipping.")
            continue

        split_ds = raw[split_name]
        n        = len(split_ds)
        print(f"\nIndexing split '{split_name}' ({n} samples) — filenames only ...")

        try:
            filenames = split_ds['image_name']
        except KeyError:
            try:
                filenames = split_ds['file_name']
            except KeyError:
                filenames = [None] * n

        meta_rows = []
        skipped   = 0
        for idx, filename in enumerate(filenames):
            if not filename:
                skipped += 1
                continue
            meta = parse_filename(filename)
            if meta is None:
                skipped += 1
                continue
            meta_rows.append({
                'split_index':     idx,
                'image_name':      filename,
                'image_id':        meta['id'],
                'beaker_capacity': meta['beaker_capacity'],
                'liquid_volume':   meta['liquid_volume'],
                'background':      meta['background'],
                'viewpoint':       meta['viewpoint'],
            })

        total_skip += skipped
        print(f"  Indexed : {len(meta_rows)}")
        print(f"  Skipped : {skipped}")
        splits_out[split_name] = {'meta': meta_rows, 'raw_split': split_ds}

    # Manual fallback split
    if len(splits_out) < 2:
        print("\nPerforming manual 70/15/15 split ...")
        key      = list(splits_out.keys())[0]
        all_meta = splits_out[key]['meta']
        raw_sp   = splits_out[key]['raw_split']
        indices  = list(range(len(all_meta)))
        tv_idx, test_idx = train_test_split(
            indices, test_size=TEST_RATIO,
            random_state=DATASET_SPLIT_SEED, shuffle=True)
        train_idx, val_idx = train_test_split(
            tv_idx, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
            random_state=DATASET_SPLIT_SEED, shuffle=True)
        splits_out = {
            'train':      {'meta': [all_meta[i] for i in train_idx], 'raw_split': raw_sp},
            'validation': {'meta': [all_meta[i] for i in val_idx],   'raw_split': raw_sp},
            'test':       {'meta': [all_meta[i] for i in test_idx],  'raw_split': raw_sp},
        }

    total = sum(len(v['meta']) for v in splits_out.values())
    print(f"\n{'='*50}")
    print(f"Dataset ready  —  {total} usable samples")
    for k, v in splits_out.items():
        print(f"  {k:<12}: {len(v['meta']):>5} samples")
    print(f"  ({total_skip} samples skipped)")
    print(f"{'='*50}")
    return splits_out


# ── Save test data ────────────────────────────────────────────────────────────

def save_test_data(splits: dict, output_dir: Path = TEST_DATA_DIR) -> Path:
    """Save test images + metadata JSON lazily (one image at a time)."""
    test_info  = splits['test']
    meta_rows  = test_info['meta']
    raw_split  = test_info['raw_split']

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(meta_rows)} test images to {output_dir} ...")

    metadata = []
    for row in meta_rows:
        sample = raw_split[row['split_index']]
        img    = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert('RGB')
        else:
            img = img.convert('RGB')
        img.save(images_dir / row['image_name'])
        metadata.append({
            'image_name':      row['image_name'],
            'image_id':        row['image_id'],
            'beaker_capacity': row['beaker_capacity'],
            'liquid_volume':   row['liquid_volume'],
            'background':      row['background'],
            'viewpoint':       row['viewpoint'],
        })

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "README.txt", 'w') as f:
        f.write("Beaker Volume Detection – Test Dataset\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Total samples : {len(meta_rows)}\n")
        f.write("Images folder : images/\n")
        f.write("Metadata file : metadata.json\n")
    print(f"  Saved {len(meta_rows)} images + metadata.json + README.txt")
    return output_dir


# ── Volume extraction ─────────────────────────────────────────────────────────

def extract_volume_from_text(text: str) -> Tuple[Optional[int], Optional[float]]:
    """Extract beaker capacity and liquid volume from model output text."""
    text = text.strip()
    b = re.search(r'beaker\s*[:\-]?\s*(\d+)\s*ml',      text, re.IGNORECASE)
    v = re.search(r'volume\s*[:\-]?\s*(\d+\.?\d*)\s*ml', text, re.IGNORECASE)
    return (int(b.group(1)) if b else None,
            float(v.group(1)) if v else None)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class BeakerDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset.
    Images fetched on demand inside __getitem__.

    LABEL STRATEGY:
      Florence-2 : labels are the answer tokens padded to MAX_LENGTH
      Qwen2-VL   : labels = full input_ids with prompt positions masked -100
                   (same length as input_ids — required by Qwen2-VL)
    """

    def __init__(self, split_info: dict, processor,
                 prompt_template: str, model_type: str = "florence"):
        self.meta       = split_info['meta']
        self.raw_split  = split_info['raw_split']
        self.processor  = processor
        self.prompt     = prompt_template
        self.model_type = model_type

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta[idx]

        # Load image lazily
        sample = self.raw_split[row['split_index']]
        image  = sample['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        beaker_cap  = row['beaker_capacity']
        liquid_vol  = row['liquid_volume']
        target_text = f"Beaker: {int(beaker_cap)}mL, Volume: {liquid_vol}mL"

        # ── Florence-2 ────────────────────────────────────────────────────
        if self.model_type == "florence":
            inputs = self.processor(
                text=self.prompt, images=image, return_tensors="pt"
            )
            targets = self.processor.tokenizer(
                text=target_text,
                return_tensors="pt",
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True
            )
            return {
                'pixel_values':    inputs['pixel_values'].squeeze(0),
                'input_ids':       inputs['input_ids'].squeeze(0),
                'attention_mask':  inputs.get(
                    'attention_mask',
                    torch.ones_like(inputs['input_ids'])
                ).squeeze(0),
                'labels':          targets['input_ids'].squeeze(0),
                'beaker_capacity': beaker_cap,
                'liquid_volume':   liquid_vol,
                'image_name':      row['image_name'],
            }

        # ── Qwen2-VL ──────────────────────────────────────────────────────
        else:
            from qwen_vl_utils import process_vision_info

            # Build full conversation (prompt + answer) - encode ONCE only
            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  self.prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target_text}]
                }
            ]

            full_text    = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(full_messages)

            # Encode full sequence ONCE
            full_enc = self.processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt"
            )

            input_ids      = full_enc['input_ids'].squeeze(0)       # [L]
            attention_mask = full_enc['attention_mask'].squeeze(0)   # [L]

            # Find answer length by tokenising ONLY the answer text (no image)
            # This avoids double-encoding the image
            answer_ids = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt"
            )['input_ids'].squeeze(0)                                # [A]
            answer_len = answer_ids.shape[0]

            # Labels: -100 for all prompt tokens, answer tokens for loss
            # input_ids and labels are guaranteed the SAME length [L]
            labels = torch.full_like(input_ids, -100)
            labels[-answer_len:] = input_ids[-answer_len:]

            return {
                'input_ids':       input_ids,
                'attention_mask':  attention_mask,
                'pixel_values':    full_enc.get('pixel_values',   torch.tensor([])),
                'image_grid_thw':  full_enc.get('image_grid_thw', torch.tensor([])),
                'labels':          labels,               # same shape as input_ids ✓
                'beaker_capacity': beaker_cap,
                'liquid_volume':   liquid_vol,
                'image_name':      row['image_name'],
            }

# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data loading (lazy mode)...\n")
    splits = load_and_split_dataset()

    print("\nSample metadata (no image loaded yet):")
    row = splits['train']['meta'][0]
    for k, v in row.items():
        if k != 'split_index':
            print(f"  {k:<18}: {v}")

    print("\nLazy image access test ...")
    sample = splits['train']['raw_split'][row['split_index']]
    img    = sample['image']
    print(f"  Image size: {img.size if hasattr(img,'size') else type(img)}")

    if not (TEST_DATA_DIR / "metadata.json").exists():
        save_test_data(splits)

    print("\n✓ data_utils.py OK")
    print(f"  Train      : {len(splits['train']['meta'])}")
    print(f"  Validation : {len(splits['validation']['meta'])}")
    print(f"  Test       : {len(splits['test']['meta'])}")