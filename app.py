"""
app.py  —  Beaker Volume Detection Demo
────────────────────────────────────────
Loads Florence-2 and Qwen2-VL directly from HuggingFace Hub
(no local training required) and exposes a Gradio interface.

Deploy as a HuggingFace Space:
  1. Create a new Space (SDK: Gradio, hardware: CPU or T4)
  2. Upload this file + requirements_space.txt
  3. Set HF_FLORENCE_REPO and HF_QWEN_REPO as Space Secrets (or edit below)
"""

import os
import re
import time
import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — edit these OR set them as HuggingFace Space Secrets
# ─────────────────────────────────────────────────────────────────────────────

FLORENCE_REPO = os.getenv("HF_FLORENCE_REPO", "yusufbukarmaina/beaker-florence2")
QWEN_REPO     = os.getenv("HF_QWEN_REPO",     "yusufbukarmaina/beaker-qwen2vl")

FLORENCE_PROMPT = "What is the beaker capacity and liquid volume in this image?"

QWEN_PROMPT = (
    "You are a precise measurement assistant. Analyze this beaker image and determine:\n"
    "1. The beaker capacity (100mL or 250mL)\n"
    "2. The current liquid volume in mL\n\n"
    "Respond in the format: \"Beaker: {capacity}mL, Volume: {volume}mL\""
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

# ─────────────────────────────────────────────────────────────────────────────
# Lazy model loading  (loaded once on first use to keep startup fast)
# ─────────────────────────────────────────────────────────────────────────────

_florence_model     = None
_florence_processor = None
_qwen_model         = None
_qwen_processor     = None


def _load_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        print(f"Loading Florence-2 from {FLORENCE_REPO} …")
        _florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_REPO, trust_remote_code=True
        )
        _florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_REPO,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        ).to(DEVICE).eval()
        print("Florence-2 ready.")
    return _florence_model, _florence_processor


def _load_qwen():
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        print(f"Loading Qwen2-VL from {QWEN_REPO} …")
        from transformers import Qwen2VLForConditionalGeneration
        _qwen_processor = AutoProcessor.from_pretrained(
            QWEN_REPO, trust_remote_code=True
        )
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_REPO,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        ).to(DEVICE).eval()
        print("Qwen2-VL ready.")
    return _qwen_model, _qwen_processor


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_prediction(text: str) -> dict:
    """Extract capacity and volume numbers from model output."""
    result = {"raw": text, "capacity_ml": None, "volume_ml": None}
    cap_match = re.search(r"[Bb]eaker[:\s]+(\d+)\s*mL", text)
    vol_match  = re.search(r"[Vv]olume[:\s]+(\d+(?:\.\d+)?)\s*mL", text)
    if cap_match:
        result["capacity_ml"] = int(cap_match.group(1))
    if vol_match:
        result["volume_ml"] = float(vol_match.group(1))
    return result


def run_florence(image: Image.Image) -> tuple[str, dict]:
    """Run Florence-2 inference on a single PIL image."""
    model, processor = _load_florence()
    image = image.convert("RGB")
    inputs = processor(
        text=FLORENCE_PROMPT,
        images=image,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=64,
            num_beams=3,
            early_stopping=True,
        )

    raw_output = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    return raw_output, _parse_prediction(raw_output)


def run_qwen(image: Image.Image) -> tuple[str, dict]:
    """Run Qwen2-VL inference on a single PIL image."""
    from qwen_vl_utils import process_vision_info

    model, processor = _load_qwen()
    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  QWEN_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
        )

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    raw_output = processor.batch_decode(
        trimmed, skip_special_tokens=True
    )[0].strip()

    return raw_output, _parse_prediction(raw_output)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict(image, model_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio.
    Returns (formatted_result, raw_output, inference_time_str)
    """
    if image is None:
        return "Please upload an image.", "", ""

    pil_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image

    t0 = time.time()
    try:
        if model_choice == "Florence-2":
            raw, parsed = run_florence(pil_image)
        elif model_choice == "Qwen2-VL":
            raw, parsed = run_qwen(pil_image)
        else:
            return "Unknown model selected.", "", ""
    except Exception as e:
        return f"Error during inference: {e}", "", ""

    elapsed = time.time() - t0

    # Format the structured result
    if parsed["capacity_ml"] and parsed["volume_ml"] is not None:
        formatted = (
            f"🧪 Beaker Capacity : **{parsed['capacity_ml']} mL**\n"
            f"💧 Liquid Volume   : **{parsed['volume_ml']} mL**\n"
            f"📊 Fill Level      : **{parsed['volume_ml'] / parsed['capacity_ml'] * 100:.1f}%**"
        )
    else:
        formatted = f"⚠️ Could not parse structured output.\n\nRaw: {raw}"

    return formatted, raw, f"⏱ {elapsed:.2f} s  ({model_choice} on {DEVICE.upper()})"


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Beaker Volume Detection",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:

        gr.Markdown(
            """
# 🧪 Beaker Volume Detection
Upload an image of a beaker and let the model predict the **capacity** and **liquid volume**.

Two fine-tuned vision-language models are available:
- **Florence-2** — Microsoft's compact vision-language model
- **Qwen2-VL** — Alibaba's 2B vision-language model

> Both models were fine-tuned on the [Beakers1](https://huggingface.co/datasets/yusufbukarmaina/Beakers1) dataset.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Beaker Image",
                    type="pil",
                    height=350,
                )
                model_choice = gr.Radio(
                    choices=["Florence-2", "Qwen2-VL"],
                    value="Florence-2",
                    label="Select Model",
                )
                run_btn = gr.Button("🔍 Detect Volume", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_output = gr.Markdown(label="Result")
                raw_output    = gr.Textbox(
                    label="Raw Model Output",
                    lines=3,
                    interactive=False,
                )
                time_output   = gr.Textbox(
                    label="Inference Info",
                    interactive=False,
                )

        run_btn.click(
            fn=predict,
            inputs=[image_input, model_choice],
            outputs=[result_output, raw_output, time_output],
        )

        # Allow pressing Enter on the image too
        image_input.change(
            fn=lambda img, m: predict(img, m),
            inputs=[image_input, model_choice],
            outputs=[result_output, raw_output, time_output],
        )

        gr.Markdown(
            """
---
### 📁 Expected filename format (for reference)
`{id}_{capacity}ml_v{volume}ml_{background}_{viewpoint}.jpg`
Example: `1284_250ml_v74ml_c_f.jpg` → 250 mL beaker, 74 mL liquid, controlled background, front view.
"""
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
