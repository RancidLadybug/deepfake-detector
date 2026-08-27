"""
detect.py
---------
Simple proof-of-concept image authenticity checker.

Uses ONE pretrained Hugging Face model that natively classifies images into
three categories directly (no combining logic needed):

    prithivMLmods/AI-vs-Deepfake-vs-Real
    -> ViT-based classifier trained specifically for: Artificial (AI-generated),
       Deepfake (manipulated), or Real.
    -> Self-reported accuracy on its own test set: ~97.5% overall.

IMPORTANT LIMITATION (be honest with yourself about this): like any
pretrained classifier, its accuracy depends heavily on how similar your
test image is to what it was trained on. It is a community-trained model
without a published paper/dataset audit, so treat results as a screening
signal, not a certified verdict — especially against very recent or
high-quality generators/deepfakes it may not have seen examples of.

Usage:
    python detect.py test_images/example.jpg
    python detect.py test_images/example.jpg --json
    python detect.py test_images/example.jpg --no-save
"""

import argparse
import json
import os
import sys
from datetime import datetime

MODEL_ID = "prithivMLmods/AI-vs-Deepfake-vs-Real"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Maps the model's raw labels to a clearer verdict string
LABEL_TO_VERDICT = {
    "artificial": "Likely AI-GENERATED",
    "deepfake": "Likely DEEPFAKE / Manipulated",
    "real": "Likely REAL",
}


def load_model():
    """Load the pretrained pipeline. Downloads weights on first run
    (requires internet) and caches them locally afterwards
    (~/.cache/huggingface on Windows: C:\\Users\\<you>\\.cache\\huggingface)."""
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        print("ERROR: Required package not installed.")
        print(f"Missing module: {e.name}")
        print("Fix: activate your environment and run:")
        print("    pip install torch transformers pillow")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"Loading model on {device_name}... (first run downloads the weights, may take a minute)")

    try:
        pipe = pipeline("image-classification", model=MODEL_ID, device=device)
    except OSError as e:
        print("ERROR: Could not download or load the model.")
        print("This is usually a network problem or Hugging Face Hub being unreachable.")
        print(f"Details: {e}")
        sys.exit(1)

    return pipe


def classify_image(image_path, pipe):
    from PIL import Image, UnidentifiedImageError

    if not os.path.isfile(image_path):
        print(f"ERROR: File not found: {image_path}")
        sys.exit(1)

    try:
        image = Image.open(image_path)
        image.verify()  # check it's a valid image file
        image = Image.open(image_path).convert("RGB")  # reopen after verify()
    except UnidentifiedImageError:
        print(f"ERROR: '{image_path}' is not a valid/readable image file.")
        sys.exit(1)
    except OSError as e:
        print(f"ERROR: Could not open image '{image_path}'. Details: {e}")
        sys.exit(1)

    results = pipe(image)  # e.g. [{'label': 'Real', 'score': 0.91}, {'label': 'Artificial', ...}, ...]
    return results


def decide_verdict(results):
    top = max(results, key=lambda r: r["score"])
    label_lower = top["label"].lower()
    verdict = LABEL_TO_VERDICT.get(label_lower, top["label"])

    return {
        "verdict": verdict,
        "confidence": round(top["score"], 4),
        "raw_scores": {r["label"]: round(r["score"], 4) for r in results},
    }


def print_report(image_path, results, verdict):
    print("\n" + "=" * 55)
    print(" IMAGE AUTHENTICITY CHECK (proof-of-concept)")
    print("=" * 55)
    print(f" File:              {image_path}")
    print("-" * 55)
    print(" Model scores:")
    for r in sorted(results, key=lambda x: -x["score"]):
        print(f"     {r['label']:<20} {r['score']*100:6.2f}%")
    print("-" * 55)
    print(f" VERDICT:           {verdict['verdict']}")
    print(f" Confidence:        {verdict['confidence']*100:.2f}%")
    print("=" * 55)
    print(" NOTE: This is one pretrained community classifier, not a")
    print(" certified detector. Treat results as a screening signal —")
    print(" accuracy depends on how similar the image is to what the")
    print(" model was trained on. See README.md for details.")
    print("=" * 55 + "\n")


def save_result(image_path, results, verdict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{basename}_{timestamp}.json")

    payload = {
        "image": os.path.abspath(image_path),
        "timestamp": timestamp,
        "model": MODEL_ID,
        "raw_results": results,
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Proof-of-concept image authenticity checker.")
    parser.add_argument("image_path", help="Path to the image file to analyse")
    parser.add_argument("--json", action="store_true", help="Print raw JSON result to stdout as well")
    parser.add_argument("--no-save", action="store_true", help="Do not save a JSON result file to results/")
    args = parser.parse_args()

    pipe = load_model()
    results = classify_image(args.image_path, pipe)
    verdict = decide_verdict(results)

    print_report(args.image_path, results, verdict)

    if not args.no_save:
        out_path = save_result(args.image_path, results, verdict)
        print(f"Result saved to: {out_path}")

    if args.json:
        print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
