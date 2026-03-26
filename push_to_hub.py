"""
push_to_hub.py
──────────────
Pushes your trained Florence-2 and/or Qwen2-VL models to HuggingFace Hub
after training is complete.

Usage:
  python push_to_hub.py --model florence --repo yusufbukarmaina/beaker-florence2
  python push_to_hub.py --model qwen     --repo yusufbukarmaina/beaker-qwen2vl
  python push_to_hub.py --model both     --repo-florence yusufbukarmaina/beaker-florence2 \
                                         --repo-qwen     yusufbukarmaina/beaker-qwen2vl

Requirements:
  pip install huggingface_hub
  huggingface-cli login    ← run this once before pushing
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# ── Import project config ─────────────────────────────────────────────────────
from config import (
    FLORENCE_OUTPUT_DIR, QWEN_OUTPUT_DIR,
    FLORENCE_MODEL_NAME, QWEN_MODEL_NAME,
    FLORENCE_CONFIG, QWEN_CONFIG,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_repo(repo_id: str, private: bool, api: HfApi) -> None:
    """Create repo on Hub if it doesn't exist yet."""
    try:
        create_repo(repo_id, private=private, exist_ok=True)
        print(f"  ✓ Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Could not create repo '{repo_id}': {e}")
        sys.exit(1)


def push_model(
    local_dir: Path,
    repo_id: str,
    model_label: str,
    private: bool,
    api: HfApi,
) -> None:
    """Load a fine-tuned model + processor from disk and push to Hub."""
    final_model_dir = local_dir / "final_model"

    if not final_model_dir.exists():
        print(f"\n⚠  '{final_model_dir}' not found – have you finished training {model_label}?")
        return

    print(f"\n{'='*60}")
    print(f"Pushing {model_label} → {repo_id}")
    print(f"{'='*60}")

    ensure_repo(repo_id, private=private, api=api)

    # ── Load processor ────────────────────────────────────────────────────
    print("  Loading processor …")
    processor = AutoProcessor.from_pretrained(
        str(final_model_dir), trust_remote_code=True
    )

    # ── Load model (merge LoRA weights into base before pushing) ──────────
    print("  Loading model (merging LoRA adapters) …")
    try:
        from peft import PeftModel

        if model_label == "Florence-2":
            base = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_NAME,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, str(final_model_dir))
        model = model.merge_and_unload()          # fuses LoRA → single weights
        print("  ✓ LoRA merged into base model")

    except Exception as e:
        # Fallback: push adapter-only checkpoint (still usable from Hub)
        print(f"  ⚠  Merge failed ({e}), pushing adapter checkpoint instead …")
        api.upload_folder(
            folder_path=str(final_model_dir),
            repo_id=repo_id,
            commit_message=f"Upload {model_label} fine-tuned adapter",
        )
        _push_training_summary(local_dir, repo_id, api)
        return

    # ── Push merged model ─────────────────────────────────────────────────
    print("  Pushing model weights …")
    model.push_to_hub(repo_id, commit_message=f"Upload merged {model_label} model")

    print("  Pushing processor / tokenizer …")
    processor.push_to_hub(repo_id, commit_message=f"Upload {model_label} processor")

    # ── Push training summary ─────────────────────────────────────────────
    _push_training_summary(local_dir, repo_id, api)

    print(f"\n  ✓ {model_label} pushed successfully!")
    print(f"    URL: https://huggingface.co/{repo_id}")


def _push_training_summary(local_dir: Path, repo_id: str, api: HfApi) -> None:
    summary_path = local_dir / "training_summary.json"
    if summary_path.exists():
        print("  Pushing training_summary.json …")
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="training_summary.json",
            repo_id=repo_id,
            commit_message="Add training summary",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Push trained beaker-detection models to HuggingFace Hub"
    )
    p.add_argument(
        "--model",
        choices=["florence", "qwen", "both"],
        required=True,
        help="Which model to push",
    )
    p.add_argument(
        "--repo",
        default=None,
        help="Hub repo ID (used when --model is 'florence' or 'qwen'). "
             "Format: username/repo-name",
    )
    p.add_argument(
        "--repo-florence",
        default=None,
        dest="repo_florence",
        help="Hub repo ID for Florence-2 (used when --model both)",
    )
    p.add_argument(
        "--repo-qwen",
        default=None,
        dest="repo_qwen",
        help="Hub repo ID for Qwen2-VL (used when --model both)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create private repositories (default: public)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    api  = HfApi()

    # Validate token
    try:
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception:
        print("✗ Not logged in. Run:  huggingface-cli login")
        sys.exit(1)

    if args.model == "florence":
        if not args.repo:
            print("✗ Please provide --repo for Florence-2")
            sys.exit(1)
        push_model(FLORENCE_OUTPUT_DIR, args.repo, "Florence-2", args.private, api)

    elif args.model == "qwen":
        if not args.repo:
            print("✗ Please provide --repo for Qwen2-VL")
            sys.exit(1)
        push_model(QWEN_OUTPUT_DIR, args.repo, "Qwen2-VL", args.private, api)

    elif args.model == "both":
        if not args.repo_florence or not args.repo_qwen:
            print("✗ When --model both, provide both --repo-florence and --repo-qwen")
            sys.exit(1)
        push_model(FLORENCE_OUTPUT_DIR, args.repo_florence, "Florence-2", args.private, api)
        push_model(QWEN_OUTPUT_DIR,     args.repo_qwen,     "Qwen2-VL",   args.private, api)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
