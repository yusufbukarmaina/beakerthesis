"""
Evaluation script for Florence-2 and Qwen2-VL models.
Computes MAE, RMSE, R² on the test split.
Compatible with the lazy-loading data_utils format.
"""
import sys
import importlib
import json
import re
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Optional, Tuple

# Force reload to avoid stale kernel cache
for mod_name in ['config', 'data_utils']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from config import (
    FLORENCE_MODEL_NAME, QWEN_MODEL_NAME,
    FLORENCE_OUTPUT_DIR, QWEN_OUTPUT_DIR,
    RESULTS_DIR, FLORENCE_PROMPT_TEMPLATE, QWEN_PROMPT_TEMPLATE
)
from data_utils import load_and_split_dataset, extract_volume_from_text

sns.set_style("whitegrid")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image(split_info: dict, idx: int) -> Image.Image:
    """Fetch one image lazily from split_info using meta index."""
    row    = split_info['meta'][idx]
    sample = split_info['raw_split'][row['split_index']]
    img    = sample['image']
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert('RGB')
    return img.convert('RGB')


def get_meta(split_info: dict, idx: int) -> dict:
    """Return metadata row for index idx."""
    return split_info['meta'][idx]


# ── Model Evaluator ───────────────────────────────────────────────────────────

class ModelEvaluator:
    """Loads a trained model and evaluates it on the test split."""

    def __init__(self, model_path: str, model_type: str):
        self.model_path = Path(model_path)
        self.model_type = model_type.lower()
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\nLoading {model_type.upper()} from {model_path}")

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        if self.model_type == "florence":
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME,
                torch_dtype=torch.bfloat16,  # Match training dtype
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_path))

        else:  # qwen
            from transformers import Qwen2VLForConditionalGeneration
            from peft import PeftModel
            # Constrain image resolution (same as training)
            self.processor.image_processor.min_pixels = 256 * 28 * 28
            self.processor.image_processor.max_pixels = 512 * 28 * 28
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_path))

        self.model.eval()
        print(f"  ✓ Loaded on {self.device}")

    def predict_one(self, image: Image.Image) -> Tuple[Optional[int], Optional[float]]:
        """Run inference on a single PIL image. Returns (beaker_cap, liquid_vol)."""

        if self.model_type == "florence":
            inputs = self.processor(
                text=FLORENCE_PROMPT_TEMPLATE,
                images=image,
                return_tensors="pt"
            )
            # Convert to bfloat16 to match training dtype
            inputs = {
                k: v.to(self.device, dtype=torch.bfloat16) if v.dtype == torch.float32 else v.to(self.device)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=50, num_beams=3, do_sample=False
                )
            text_out = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        else:  # qwen
            from qwen_vl_utils import process_vision_info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text":  QWEN_PROMPT_TEMPLATE}
                    ]
                }
            ]
            text_input   = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=50, num_beams=3, do_sample=False
                )
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs['input_ids'], generated_ids)
            ]
            text_out = self.processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        beaker_cap, liquid_vol = extract_volume_from_text(text_out)
        return (beaker_cap or 0), (liquid_vol or 0.0), text_out

    def evaluate(self, test_split_info: dict) -> Dict:
        """
        Evaluate on the full test split.

        Args:
            test_split_info: splits['test'] dict from load_and_split_dataset()
                             has keys 'meta' and 'raw_split'
        Returns:
            results dict with metrics and per-sample predictions
        """
        n = len(test_split_info['meta'])
        print(f"\nEvaluating {self.model_type.upper()} on {n} test samples ...")

        details = []

        for idx in tqdm(range(n), desc=f"Evaluating {self.model_type}"):
            row   = get_meta(test_split_info, idx)
            image = get_image(test_split_info, idx)

            true_beaker = row['beaker_capacity']
            true_volume = row['liquid_volume']

            try:
                pred_beaker, pred_volume, raw_text = self.predict_one(image)
            except Exception as e:
                print(f"\n  Warning: failed on sample {idx} ({row['image_name']}): {e}")
                pred_beaker, pred_volume, raw_text = 0, 0.0, ""

            details.append({
                'image_name':   row['image_name'],
                'background':   row['background'],
                'viewpoint':    row['viewpoint'],
                'true_beaker':  true_beaker,
                'true_volume':  float(true_volume),
                'pred_beaker':  pred_beaker,
                'pred_volume':  float(pred_volume),
                'beaker_error': abs(pred_beaker - true_beaker),
                'volume_error': abs(float(pred_volume) - float(true_volume)),
                'raw_text':     raw_text,
            })

        # ── Compute metrics ───────────────────────────────────────────────
        true_vols = [d['true_volume']  for d in details]
        pred_vols = [d['pred_volume']  for d in details]
        true_caps = [d['true_beaker']  for d in details]
        pred_caps = [d['pred_beaker']  for d in details]

        mae  = mean_absolute_error(true_vols, pred_vols)
        rmse = float(np.sqrt(mean_squared_error(true_vols, pred_vols)))
        r2   = r2_score(true_vols, pred_vols)
        beaker_acc = float(np.mean([t == p for t, p in zip(true_caps, pred_caps)]))

        results = {
            'model_type':  self.model_type,
            'num_samples': n,
            'liquid_volume_metrics': {
                'mae':  float(mae),
                'rmse': float(rmse),
                'r2':   float(r2),
            },
            'beaker_capacity_metrics': {
                'accuracy': beaker_acc,
            },
            'predictions': details,
        }

        print(f"\n{'='*55}")
        print(f"Results — {self.model_type.upper()}")
        print(f"{'='*55}")
        print(f"  MAE  : {mae:.2f} mL")
        print(f"  RMSE : {rmse:.2f} mL")
        print(f"  R²   : {r2:.4f}")
        print(f"  Beaker capacity accuracy: {beaker_acc*100:.1f}%")
        print(f"{'='*55}")

        return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(florence_res: Dict, qwen_res: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, label in zip(
        axes,
        ['mae', 'rmse', 'r2'],
        ['MAE (mL)', 'RMSE (mL)', 'R² Score']
    ):
        fv = florence_res['liquid_volume_metrics'][metric]
        qv = qwen_res['liquid_volume_metrics'][metric]
        bars = ax.bar(['Florence-2', 'Qwen2-VL'], [fv, qv],
                      color=['#3498db', '#e74c3c'], alpha=0.75)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, [fv, qv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.suptitle('Model Comparison — Liquid Volume Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ metrics_comparison.png")

    # 2. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, res, name, color in [
        (axes[0], florence_res, 'Florence-2', '#3498db'),
        (axes[1], qwen_res,     'Qwen2-VL',   '#e74c3c'),
    ]:
        tv = [d['true_volume'] for d in res['predictions']]
        pv = [d['pred_volume'] for d in res['predictions']]
        ax.scatter(tv, pv, alpha=0.4, color=color, s=20)
        lo, hi = min(tv+pv), max(tv+pv)
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect')
        ax.set_xlabel('True Volume (mL)')
        ax.set_ylabel('Predicted Volume (mL)')
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        r2 = res['liquid_volume_metrics']['r2']
        ax.text(0.05, 0.95, f'R²={r2:.4f}', transform=ax.transAxes,
                va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'prediction_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ prediction_scatter.png")

    # 3. Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, name, color in [
        (axes[0], florence_res, 'Florence-2', '#3498db'),
        (axes[1], qwen_res,     'Qwen2-VL',   '#e74c3c'),
    ]:
        errs = [d['volume_error'] for d in res['predictions']]
        ax.hist(errs, bins=25, color=color, alpha=0.7, edgecolor='black')
        mu = np.mean(errs)
        ax.axvline(mu, color='red', linestyle='--', lw=2, label=f'Mean={mu:.1f}mL')
        ax.set_xlabel('Absolute Error (mL)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} — Error Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ error_distribution.png")

    # 4. Performance by viewpoint
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, res, name in [
        (axes[0], florence_res, 'Florence-2'),
        (axes[1], qwen_res,     'Qwen2-VL'),
    ]:
        vp_err = {}
        for d in res['predictions']:
            vp_err.setdefault(d['viewpoint'], []).append(d['volume_error'])
        vps   = sorted(vp_err.keys())
        means = [np.mean(vp_err[v]) for v in vps]
        bars  = ax.bar(vps, means, color='steelblue', alpha=0.75, edgecolor='black')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Viewpoint')
        ax.set_ylabel('Mean Absolute Error (mL)')
        ax.set_title(f'{name} — By Viewpoint', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_by_viewpoint.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ performance_by_viewpoint.png")

    # 5. Performance by background
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, res, name in [
        (axes[0], florence_res, 'Florence-2'),
        (axes[1], qwen_res,     'Qwen2-VL'),
    ]:
        bg_err = {}
        for d in res['predictions']:
            bg_err.setdefault(d['background'], []).append(d['volume_error'])
        bgs   = sorted(bg_err.keys())
        means = [np.mean(bg_err[b]) for b in bgs]
        bars  = ax.bar(bgs, means, color='coral', alpha=0.75, edgecolor='black')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('Background')
        ax.set_ylabel('Mean Absolute Error (mL)')
        ax.set_title(f'{name} — By Background', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'performance_by_background.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ performance_by_background.png")


def save_comparison_table(florence_res: Dict, qwen_res: Dict, out_dir: Path) -> pd.DataFrame:
    f_preds = florence_res['predictions']
    q_preds = qwen_res['predictions']

    rows = {
        'Metric': [
            'MAE (mL)', 'RMSE (mL)', 'R² Score',
            'Beaker Accuracy (%)',
            'Mean Abs Error (mL)', 'Std Error (mL)',
            'Max Error (mL)', 'Min Error (mL)',
        ],
        'Florence-2': [
            f"{florence_res['liquid_volume_metrics']['mae']:.2f}",
            f"{florence_res['liquid_volume_metrics']['rmse']:.2f}",
            f"{florence_res['liquid_volume_metrics']['r2']:.4f}",
            f"{florence_res['beaker_capacity_metrics']['accuracy']*100:.1f}",
            f"{np.mean([d['volume_error'] for d in f_preds]):.2f}",
            f"{np.std( [d['volume_error'] for d in f_preds]):.2f}",
            f"{max(    [d['volume_error'] for d in f_preds]):.2f}",
            f"{min(    [d['volume_error'] for d in f_preds]):.2f}",
        ],
        'Qwen2-VL': [
            f"{qwen_res['liquid_volume_metrics']['mae']:.2f}",
            f"{qwen_res['liquid_volume_metrics']['rmse']:.2f}",
            f"{qwen_res['liquid_volume_metrics']['r2']:.4f}",
            f"{qwen_res['beaker_capacity_metrics']['accuracy']*100:.1f}",
            f"{np.mean([d['volume_error'] for d in q_preds]):.2f}",
            f"{np.std( [d['volume_error'] for d in q_preds]):.2f}",
            f"{max(    [d['volume_error'] for d in q_preds]):.2f}",
            f"{min(    [d['volume_error'] for d in q_preds]):.2f}",
        ],
    }
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'comparison_table.csv', index=False)

    with open(out_dir / 'comparison_table.txt', 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("MODEL COMPARISON: Florence-2 vs Qwen2-VL\n")
        f.write("=" * 65 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "=" * 65 + "\n")

    print("  ✓ comparison_table.csv / .txt")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BEAKER VOLUME DETECTION — MODEL EVALUATION")
    print("=" * 70)

    florence_path = FLORENCE_OUTPUT_DIR / "final_model"
    qwen_path     = QWEN_OUTPUT_DIR     / "final_model"

    if not florence_path.exists():
        print(f"\n❌ Florence-2 model not found at {florence_path}")
        print("   Run train_florence.py first.")
        return
    if not qwen_path.exists():
        print(f"\n❌ Qwen2-VL model not found at {qwen_path}")
        print("   Run train_qwen.py first.")
        return

    # ── Load test split ───────────────────────────────────────────────────
    print("\nLoading test split ...")
    splits     = load_and_split_dataset()
    test_split = splits['test']
    n_test     = len(test_split['meta'])
    print(f"Test samples: {n_test}")

    if n_test == 0:
        print("❌ No test samples found. Check data_utils.py.")
        return

    # ── Evaluate Florence-2 ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATING FLORENCE-2")
    print("=" * 70)
    f_eval          = ModelEvaluator(str(florence_path), "florence")
    florence_results = f_eval.evaluate(test_split)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'florence_results.json', 'w') as fp:
        # predictions list may contain non-serialisable floats — convert
        safe = {k: v for k, v in florence_results.items() if k != 'predictions'}
        safe['predictions'] = florence_results['predictions']
        json.dump(safe, fp, indent=2, default=str)
    print(f"  ✓ Saved florence_results.json")

    # ── Evaluate Qwen2-VL ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATING QWEN2-VL")
    print("=" * 70)
    q_eval       = ModelEvaluator(str(qwen_path), "qwen")
    qwen_results = q_eval.evaluate(test_split)

    with open(RESULTS_DIR / 'qwen_results.json', 'w') as fp:
        json.dump(qwen_results, fp, indent=2, default=str)
    print(f"  ✓ Saved qwen_results.json")

    # ── Plots & table ─────────────────────────────────────────────────────
    print("\nGenerating comparison visualisations ...")
    plot_comparison(florence_results, qwen_results, RESULTS_DIR)

    print("\nGenerating comparison table ...")
    df = save_comparison_table(florence_results, qwen_results, RESULTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))

    f_mae = florence_results['liquid_volume_metrics']['mae']
    q_mae = qwen_results['liquid_volume_metrics']['mae']
    print("\n" + "=" * 70)
    if f_mae <= q_mae:
        pct = (q_mae - f_mae) / q_mae * 100
        print(f"🏆 Florence-2 wins!  MAE {f_mae:.2f} vs {q_mae:.2f}  ({pct:.1f}% better)")
    else:
        pct = (f_mae - q_mae) / f_mae * 100
        print(f"🏆 Qwen2-VL wins!    MAE {q_mae:.2f} vs {f_mae:.2f}  ({pct:.1f}% better)")
    print("=" * 70)

    print(f"\n✓ All results saved to: {RESULTS_DIR}")
    print("\nNext: python3 demo.py")


if __name__ == "__main__":
    main()