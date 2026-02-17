"""
Training script for Qwen2-VL on beaker volume detection.
Self-contained collate that enforces equal sizes defensively.
"""
import os
import sys
import importlib
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ── Force reload config and data_utils from disk ─────────────────────────────
# This prevents stale cached module versions in Jupyter notebooks
for mod_name in ['config', 'data_utils']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from config import *
from data_utils import load_and_split_dataset, save_test_data
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from PIL import Image

torch.manual_seed(DATASET_SPLIT_SEED)
np.random.seed(DATASET_SPLIT_SEED)


# ── Inline Qwen Dataset — no dependency on data_utils.BeakerDataset ──────────

class QwenBeakerDataset(Dataset):
    """
    Self-contained Qwen2-VL dataset.
    Labels are always the SAME shape as input_ids — guaranteed.
    Strategy: encode full sequence once, mask all but last answer_len tokens.
    """

    def __init__(self, split_info: dict, processor, prompt: str):
        self.meta      = split_info['meta']
        self.raw_split = split_info['raw_split']
        self.processor = processor
        self.prompt    = prompt

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

        # Full conversation: user (image + question) + assistant (answer)
        messages = [
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

        # Render to text and encode ONCE
        full_text    = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        enc = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )

        input_ids      = enc['input_ids'].squeeze(0)       # [L]
        attention_mask = enc['attention_mask'].squeeze(0)  # [L]
        seq_len        = input_ids.shape[0]

        # Measure answer length via text-only tokenisation (no image re-encoding)
        answer_token_ids = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            return_tensors="pt"
        )['input_ids'].squeeze(0)
        answer_len = min(answer_token_ids.shape[0], seq_len)

        # Labels: -100 everywhere except the last answer_len positions
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        labels[-answer_len:] = input_ids[-answer_len:]

        # Sanity check — both must be [L]
        assert input_ids.shape == labels.shape, (
            f"Shape mismatch: input_ids={input_ids.shape}, labels={labels.shape}"
        )

        return {
            'input_ids':       input_ids,
            'attention_mask':  attention_mask,
            'pixel_values':    enc.get('pixel_values',   torch.tensor([])),
            'image_grid_thw':  enc.get('image_grid_thw', torch.tensor([])),
            'labels':          labels,
            'beaker_capacity': beaker_cap,
            'liquid_volume':   liquid_vol,
            'image_name':      row['image_name'],
        }


# ── Trainer class ─────────────────────────────────────────────────────────────

class Qwen2VLTrainer:

    def __init__(self, output_dir: Path = QWEN_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Qwen2-VL Beaker Volume Detection Training")
        print("=" * 60)

        # ── Load processor ────────────────────────────────────────────────
        print(f"\nLoading processor: {QWEN_MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_NAME,
            trust_remote_code=True
        )
        # Constrain image resolution → limits token count variance per sample
        self.processor.image_processor.min_pixels = 256 * 28 * 28
        self.processor.image_processor.max_pixels = 512 * 28 * 28

        # ── Load model ────────────────────────────────────────────────────
        print(f"Loading model: {QWEN_MODEL_NAME}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # ── Apply LoRA ────────────────────────────────────────────────────
        print("Applying LoRA ...")
        lora_cfg = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

        # ── Load dataset (metadata only — no images in RAM) ───────────────
        print("\nLoading dataset (lazy mode) ...")
        self.splits = load_and_split_dataset()

        # Save test images once
        if not (TEST_DATA_DIR / "metadata.json").exists():
            print("\nSaving test data for demo ...")
            save_test_data(self.splits)
        else:
            print("\nTest data already saved, skipping.")

        # ── Build PyTorch datasets ────────────────────────────────────────
        print("\nBuilding PyTorch datasets ...")
        self.train_dataset = QwenBeakerDataset(
            self.splits['train'],
            self.processor,
            QWEN_PROMPT_TEMPLATE
        )
        self.val_dataset = QwenBeakerDataset(
            self.splits['validation'],
            self.processor,
            QWEN_PROMPT_TEMPLATE
        )
        print(f"  Train      : {len(self.train_dataset)} samples")
        print(f"  Validation : {len(self.val_dataset)} samples")

        # Pad token id
        self.pad_id = (self.processor.tokenizer.pad_token_id
                       or self.processor.tokenizer.eos_token_id)

    # ── Collate ───────────────────────────────────────────────────────────
    def collate_fn(self, batch):
        """
        Pad all sequences to the longest in the batch.
        input_ids and labels are always the same length per sample,
        so a single max_len covers both.
        """
        # Each sample: input_ids.shape == labels.shape (guaranteed by dataset)
        max_len = max(b['input_ids'].shape[0] for b in batch)

        input_ids_list      = []
        attention_mask_list = []
        labels_list         = []

        for b in batch:
            seq_len = b['input_ids'].shape[0]
            pad_len = max_len - seq_len

            input_ids_list.append(
                F.pad(b['input_ids'],      (0, pad_len), value=self.pad_id)
            )
            attention_mask_list.append(
                F.pad(b['attention_mask'], (0, pad_len), value=0)
            )
            labels_list.append(
                F.pad(b['labels'],         (0, pad_len), value=-100)
            )

        result = {
            'input_ids':      torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels':         torch.stack(labels_list),
        }

        pv_list = [b['pixel_values'] for b in batch
                   if b['pixel_values'].numel() > 0]
        if pv_list:
            result['pixel_values'] = torch.cat(pv_list, dim=0)

        grid_list = [b['image_grid_thw'] for b in batch
                     if b['image_grid_thw'].numel() > 0]
        if grid_list:
            result['image_grid_thw'] = torch.cat(grid_list, dim=0)

        return result

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self):
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=QWEN_CONFIG['num_epochs'],
            per_device_train_batch_size=QWEN_CONFIG['batch_size'],
            per_device_eval_batch_size=QWEN_CONFIG['batch_size'],
            gradient_accumulation_steps=QWEN_CONFIG['gradient_accumulation_steps'],
            learning_rate=QWEN_CONFIG['learning_rate'],
            weight_decay=QWEN_CONFIG['weight_decay'],
            warmup_steps=QWEN_CONFIG['warmup_steps'],
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=QWEN_CONFIG['logging_steps'],
            eval_strategy="steps",
            eval_steps=QWEN_CONFIG['eval_steps'],
            save_strategy="steps",
            save_steps=QWEN_CONFIG['save_steps'],
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=QWEN_CONFIG['fp16'],
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="tensorboard",
            push_to_hub=False,
            dataloader_pin_memory=False,
            label_names=["labels"],
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
        )

        print("\nTraining started ...")
        result = trainer.train()

        print("\nSaving final model ...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.processor.save_pretrained(str(self.output_dir / "final_model"))

        metrics = result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        with open(self.output_dir / "training_summary.json", 'w') as f:
            json.dump({
                "model":            QWEN_MODEL_NAME,
                "train_samples":    len(self.train_dataset),
                "val_samples":      len(self.val_dataset),
                "num_epochs":       QWEN_CONFIG['num_epochs'],
                "learning_rate":    QWEN_CONFIG['learning_rate'],
                "final_train_loss": metrics.get('train_loss'),
            }, f, indent=2)

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Model saved to: {self.output_dir / 'final_model'}")
        print("=" * 60)
        return trainer


def main():
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"RAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")

    trainer = Qwen2VLTrainer()
    trainer.train()

    print("\n✓ Qwen2-VL training complete!")
    print(f"  Checkpoint : {QWEN_OUTPUT_DIR / 'final_model'}")
    print(f"  Test data  : {TEST_DATA_DIR}")
    print("\nNext: python3 evaluate.py")


if __name__ == "__main__":
    main()