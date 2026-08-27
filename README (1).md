---
title: TrueLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# TrueLens — Image Authenticity Detector (Proof of Concept)

> The YAML block above this line is required by Hugging Face Spaces to
> know how to run this app. If you're just reading this locally or running
> it on your own laptop, ignore it — it doesn't affect `detect.py` or
> `app.py` at all.

A simple, no-training-required Python prototype that analyses an image and
gives a heuristic verdict of:

- **Real**
- **AI-generated**
- **Deepfake / Manipulated**

It uses two pretrained Hugging Face models (see Section 3). No custom
training happens in this stage.

---

## 1. Anaconda Setup

**Python version:** 3.10
**Environment name:** `deepfake_env`

Run these in **Anaconda Prompt** (not the Windows default `cmd`, and not the
VS Code terminal — create the environment in Anaconda Prompt first).

```bash
conda create -n deepfake_env python=3.10 -y
conda activate deepfake_env
```

### Install PyTorch (CPU-only — recommended for a laptop proof-of-concept)

Still in Anaconda Prompt, with `deepfake_env` active:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> If you specifically have an NVIDIA GPU and already have CUDA drivers
> installed, you can instead use the CUDA build (check
> https://pytorch.org/get-started/locally/ for the current command for your
> CUDA version). For a first proof-of-concept, CPU is simpler and avoids
> version-matching headaches — inference on a single image is fast enough
> on CPU.

### Install the rest of the packages

```bash
pip install transformers pillow numpy accelerate safetensors
```

(These versions are also pinned in `requirements.txt` if you prefer:
`pip install -r requirements.txt` — but install `torch` first using the
command above, since it's not in requirements.txt on purpose.)

### Which terminal for what

| Step | Where |
|---|---|
| `conda create`, `conda activate` | Anaconda Prompt |
| `pip install ...` | Anaconda Prompt (same activated env) |
| Running `detect.py` day-to-day | Either Anaconda Prompt **or** VS Code terminal — as long as VS Code's terminal is using the `deepfake_env` interpreter (bottom-right corner of VS Code, or `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `deepfake_env`) |
| Jupyter Notebook | `conda install ipykernel -y` then `python -m ipykernel install --user --name deepfake_env`, then pick "deepfake_env" as the notebook kernel |

### Verify the installation

Run this in Anaconda Prompt (env activated):

```bash
python -c "import torch, transformers, PIL; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected: it prints version numbers with no errors. `CUDA available: False`
is fine and expected on most laptops — the script runs on CPU automatically.

---

## 2. Project Folder

```
deepfake_detector/
│
├── models/            <- not used for manual files; Hugging Face models
│                          auto-download here conceptually (see Section 3)
├── test_images/        <- put your test .jpg/.png files here
├── results/            <- JSON result files are saved here automatically
├── detect.py           <- main script
├── requirements.txt
└── README.md
```

You don't need to create or download anything into `models/` by hand for
this proof-of-concept — see Section 3 for why.

---

## 3. Model

**Update:** this project originally combined two separate binary models.
It's since been simplified to **one single model that natively classifies
into all three categories** — simpler code, one thing to reason about.

### Model — `prithivMLmods/AI-vs-Deepfake-vs-Real`
- **Architecture:** ViT-Base (Vision Transformer), based on
  `google/vit-base-patch32-224-in21k`
- **Detects natively:** `Artificial` (AI-generated), `Deepfake`
  (manipulated), or `Real` — one softmax output over all three classes
- **Self-reported accuracy on its own test set:** ~97.5% overall (per the
  model card's classification report)

### Where the weights come from / how they're loaded
Hosted on the Hugging Face Hub. `transformers.pipeline(...)` downloads the
weights automatically the first time you run `detect.py` (needs internet),
and caches them locally at:

```
C:\Users\<your-username>\.cache\huggingface\hub
```

On subsequent runs, no internet is needed. Format: PyTorch / `safetensors`
weights, loaded via `transformers.AutoModelForImageClassification`
internally (the `pipeline` helper handles this for you).

### Known limitation — read this before trusting a result
This is a **community-trained model** — no published paper, no independent
audit of its training data, and its ~97.5% figure is measured on its own
held-out test set, not on images you'll actually throw at it. In practice
that means:
- It can still miss AI images from generators or deepfake techniques not
  well represented in its training data — especially very recent ones.
- A high confidence score reflects how the image compares to *what this
  model has seen before*, not an absolute truth.
- Treat every result as a **screening signal**, not a certified verdict.
  If something matters (legal, journalistic, high-stakes decisions), that
  needs human review and ideally multiple independent checks — not one
  pretrained classifier's opinion.

This is exactly the gap that fine-tuning on your own curated, up-to-date
dataset (Section 7) would help close.

---

## 4 & 5. Python Code / Ready-to-use Files

`detect.py`, `requirements.txt`, and this `README.md` are provided as
complete, runnable files (see the project folder). Run it like this:

```bash
conda activate deepfake_env
cd path\to\deepfake_detector
python detect.py test_images/example.jpg
```

Optional flags:

```bash
python detect.py test_images/example.jpg --json
python detect.py test_images/example.jpg --ai-threshold 0.6 --deepfake-threshold 0.6
python detect.py test_images/example.jpg --no-save
```

What it does:
1. Loads both pretrained pipelines (CPU or GPU, auto-detected).
2. Opens and validates your image.
3. Runs both classifiers.
4. Applies a simple decision rule (deepfake signal checked first, then
   AI-generation signal, else "Real").
5. Prints a formatted report with both raw model scores and the final
   verdict + confidence.
6. Saves a JSON file to `results/`.

---

## 6. Testing

### Good test images to try
- **A real, unedited phone photo** — should classify as *Likely REAL*.
- **An AI-generated image** (e.g. from Midjourney, DALL-E, Stable
  Diffusion) — should classify as *Likely AI-GENERATED* (accuracy varies
  by generator, see limitation above).
- **A face-swap / deepfake sample image** (search "deepfake sample dataset"
  for publicly available test sets, e.g. Kaggle's "deepfake and real
  images" dataset) — should classify as *Likely DEEPFAKE*.
- **A real face photo (selfie)** — good sanity check that Model B doesn't
  over-flag ordinary photos as fake.

### What success looks like
```
=======================================================
 IMAGE AUTHENTICITY CHECK (proof-of-concept)
=======================================================
 File:              test_images/example.jpg
-------------------------------------------------------
 Model A - AI-generated vs Human (general images):
     human                 91.20%
     AI-generated           8.80%
 Model B - Deepfake vs Real (face-specific):
     Real                  95.10%
     Fake                   4.90%
-------------------------------------------------------
 VERDICT:           Likely REAL
 Confidence:        95.10%
 Basis:             both models agree / no strong fake or AI signal
=======================================================
Result saved to: results/example_20260827_101500.json
```

### Common errors and fixes

**`ModuleNotFoundError: No module named 'transformers'` (or torch/PIL)**
→ You're not in the right environment, or a package didn't install.
Run `conda activate deepfake_env` then re-run the pip install commands in
Section 1. In VS Code, confirm the interpreter is set to `deepfake_env`.

**CUDA / PyTorch errors (e.g. `CUDA error`, `no kernel image is available`)**
→ You likely installed a CUDA build of torch without a matching GPU/driver
setup. Fix: uninstall and reinstall the CPU build:
```bash
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
The script auto-falls-back to CPU if `torch.cuda.is_available()` is False,
so CPU-only is generally the simplest path for a laptop prototype.

**"model not found" / repo not found errors**
→ Check your internet connection — the first run needs to download the
models. Also double check you haven't typo'd a model ID if you modify the
script. If you're behind a corporate proxy/firewall blocking
huggingface.co, that will also cause this.

**Incompatible Python/package version errors**
→ Confirm `python --version` inside the activated env shows 3.10.x. If you
created the env with a different Python version by mistake, delete and
recreate it:
```bash
conda deactivate
conda env remove -n deepfake_env
conda create -n deepfake_env python=3.10 -y
```

**Image loading errors ("not a valid/readable image file")**
→ The file is corrupted, not actually an image, or an unsupported format.
Try re-saving it as a standard `.jpg` or `.png`. HEIC files from iPhones
often need converting first (Pillow doesn't read HEIC by default).

---

## 8. Local Browser App (app.py)

A simple drag-and-drop web interface that runs entirely on your laptop,
built with Streamlit. It reuses the exact same logic as `detect.py` — no
duplicated code, same models, same decision rule.

### Install (one extra package)

```bash
conda activate deepfake_env
pip install streamlit
```

(Already included if you installed everything with `pip install -r requirements.txt`.)

### Run it

```bash
cd path\to\deepfake_detector
streamlit run app.py
```

This opens a browser tab automatically at `http://localhost:8501`. Upload
an image with the file picker, and it will show the same verdict, raw
scores, and confidence as the command-line version — plus adjustable
threshold sliders in the sidebar.

This is **local only** — it's not reachable from the internet or other
devices, just your own browser talking to a process running on your
machine. Close the terminal (Ctrl+C) to stop it. If you later want it
reachable by other people, that's the "real public website" path (hosting
on something like Hugging Face Spaces, Render, etc.) — a separate step
from what's set up here.

### Common app.py-specific issues

**`ModuleNotFoundError: No module named 'streamlit'`**
→ `pip install streamlit` in the activated `deepfake_env`.

**Browser doesn't open automatically**
→ Copy the `http://localhost:8501` URL Streamlit prints in the terminal
into your browser manually.

**Port already in use**
→ `streamlit run app.py --server.port 8502` (or any free port).

---

## 7. Training — Later (not implemented yet)

Once the prototype above is confirmed working, fine-tuning your own model
would look roughly like this:

1. **Dataset** — collect/label images into three folders or a CSV:
   `real/`, `ai_generated/`, `deepfake/`.
2. **Split** — typically 70% train / 15% validation / 15% test, stratified
   so all three classes are represented in each split.
3. **Preprocess** — resize to the model's expected input size (224×224 for
   these ViT models), normalize pixel values, optionally augment
   (flips, crops, compression artifacts to mimic real-world sharing).
4. **Fine-tune** — start from a pretrained backbone (e.g. ViT-Base) and
   retrain the final classification head — and optionally more layers — on
   your 3-class dataset using `transformers.Trainer` or a plain PyTorch
   training loop.
5. **Evaluate** — accuracy, precision/recall/F1 per class, and a confusion
   matrix on the held-out test set. Pay special attention to how it handles
   images from AI generators/deepfake methods not seen in training.
6. **Save the trained model** — `model.save_pretrained("models/my_model")`.
7. **Connect it to `detect.py`** — change `AI_MODEL_ID` /
   `DEEPFAKE_MODEL_ID` (or add a third pipeline) to point at your local
   `models/my_model` folder path instead of a Hugging Face Hub ID.

We don't need to do any of this until the prototype above is confirmed
working on real test images.
