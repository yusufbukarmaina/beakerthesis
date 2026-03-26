"""
push_to_hub.py
──────────────
Pushes trained Florence-2 and/or Qwen2-VL models to HuggingFace Hub.
Token is hardcoded — no manual login needed.

Usage:
  python push_to_hub.py --model florence
  python push_to_hub.py --model qwen
  python push_to_hub.py --model both
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo, login

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  EDIT THESE TWO LINES WITH YOUR OWN VALUES
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # paste your HF write token here
HF_USERNAME = "yusufbukarmaina"                        # your HF username
# ─────────────────────────────────────────────────────────────────────────────

FLORENCE_REPO_ID = f"{HF_USERNAME}/beaker-florence2"
QWEN_REPO_ID     = f"{HF_USERNAME}/beaker-qwen2vl"

from config import (
    FLORENCE_OUTPUT_DIR, QWEN_OUTPUT_DIR,
    FLORENCE_MODEL_NAME, QWEN_MODEL_NAME,
)


def do_login():
    """Login to HuggingFace using the hardcoded token."""
    print("Logging in to HuggingFace ...")
    login(token=HF_TOKEN)
    api = HfApi(token=HF_TOKEN)
    user = api.whoami()
    print(f"✓ Logged in as: {user['name']}")
    return api


def ensure_repo(repo_id: str, private: bool, api: HfApi):
    try:
        create_repo(repo_id, private=private, exist_ok=True, token=HF_TOKEN)
        print(f"  ✓ Repo ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"  ✗ Could not create repo '{repo_id}': {e}")
        sys.exit(1)


def push_model(local_dir: Path, repo_id: str, model_label: str,
               private: bool, api: HfApi):

    final_model_dir = local_dir / "final_model"

    if not final_model_dir.exists():
        print(f"\n⚠  '{final_model_dir}' not found — has {model_label} finished training?")
        return

    print(f"\n{'='*60}")
    print(f"Pushing {model_label}  →  {repo_id}")
    print(f"{'='*60}")

    ensure_repo(repo_id, private=private, api=api)

    # ── Load processor ────────────────────────────────────────────────────
    print("  Loading processor ...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        str(final_model_dir), trust_remote_code=True
    )

    # ── Load model and merge LoRA ─────────────────────────────────────────
    print("  Loading model and merging LoRA adapters ...")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

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
        model = model.merge_and_unload()
        print("  ✓ LoRA merged successfully")

    except Exception as e:
        print(f"  ⚠  Merge failed ({e}), pushing adapter checkpoint instead ...")
        api.upload_folder(
            folder_path=str(final_model_dir),
            repo_id=repo_id,
            token=HF_TOKEN,
            commit_message=f"Upload {model_label} adapter checkpoint",
        )
        _push_summary(local_dir, repo_id, api)
        return

    # ── Push to Hub ───────────────────────────────────────────────────────
    print("  Pushing model weights (this may take a few minutes) ...")
    model.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        commit_message=f"Upload merged {model_label} fine-tuned model"
    )

    print("  Pushing processor / tokenizer ...")
    processor.push_to_hub(
        repo_id,
        token=HF_TOKEN,
        commit_message=f"Upload {model_label} processor"
    )

    _push_summary(local_dir, repo_id, api)

    print(f"\n  ✓ {model_label} pushed successfully!")
    print(f"    View at: https://huggingface.co/{repo_id}")


def _push_summary(local_dir: Path, repo_id: str, api: HfApi):
    summary_path = local_dir / "training_summary.json"
    if summary_path.exists():
        print("  Pushing training_summary.json ...")
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="training_summary.json",
            repo_id=repo_id,
            token=HF_TOKEN,
            commit_message="Add training summary",
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["florence", "qwen", "both"],
        required=True,
        help="Which model to push"
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make repos private (default: public)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    api  = do_login()

    if args.model == "florence":
        push_model(FLORENCE_OUTPUT_DIR, FLORENCE_REPO_ID,
                   "Florence-2", args.private, api)

    elif args.model == "qwen":
        push_model(QWEN_OUTPUT_DIR, QWEN_REPO_ID,
                   "Qwen2-VL", args.private, api)

    elif args.model == "both":
        push_model(FLORENCE_OUTPUT_DIR, FLORENCE_REPO_ID,
                   "Florence-2", args.private, api)
        push_model(QWEN_OUTPUT_DIR, QWEN_REPO_ID,
                   "Qwen2-VL", args.private, api)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
