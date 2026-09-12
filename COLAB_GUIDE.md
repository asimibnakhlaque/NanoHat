# COLAB TRAINING GUIDE — NanoHat-360M v2 (GREEN FLAG build)

Auditor verdict: GREEN FLAG (delta-2, all detectors at absolute zero).
Train on the certified corpus exactly as-is.

## WHAT YOU'LL TRAIN ON
- dataset/v2/train.json — 30,792 conversations (pure ox-alpha distilled lineage)
- dataset/v2/eval.json  —  1,380 conversations (held-out, capability-aware)
- (separate, NOT in this run: from_gemma_to_oxalpha/ — 15,118 certified convs.
  Optional later experiment at reduced sampling weight.)

## FILES TO UPLOAD TO COLAB
1. dataset/v2/train.json
2. dataset/v2/eval.json
3. training/colab_train_v2.py

## COLAB SETUP (T4 GPU runtime)
1. Runtime -> Change runtime type -> T4 GPU
2. Cells below, in order.

### Cell 1 — deps (2-3 min)
```python
!pip install -q unsloth
```

### Cell 2 — mount Drive + upload data
```python
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
os.makedirs('dataset/v2', exist_ok=True)
os.makedirs('training', exist_ok=True)
# If files are on Drive already, copy from there instead of manual upload:
for f in ['train.json', 'eval.json']:
    src = f'/content/drive/MyDrive/nanohat_v2_upload/{f}'
    if os.path.exists(src):
        shutil.copy(src, f'dataset/v2/{f}')
```
(Or use the Files sidebar to drag-upload train.json / eval.json /
colab_train_v2.py, then move them into place.)

### Cell 3 — PROBE (never skip; ~2 min)
```python
!python training/colab_train_v2.py --probe --train dataset/v2/train.json
```
Must print "=== PROBE PASS ===" (proves tool-role tokens are excluded from loss).

### Cell 4 — TRAIN (fresh run; ~3-5h over 3 epochs)
```python
!python training/colab_train_v2.py \
    --drive-dir /content/drive/MyDrive/nanohat_v2 \
    --train dataset/v2/train.json --eval dataset/v2/eval.json \
    --epochs 3
```
- Checkpoints hit Drive every 500 steps (adapter + optimizer state)
- If the session dies: reopen, mount Drive, re-upload colab_train_v2.py, run
  Cell 1 + Cell 2, then Cell 4 with `--resume` — it restores the newest
  checkpoint and continues exactly where it stopped.

### Cell 5 — EXPORT (merge + GGUF)
```python
from training.merge_adapter import export_merged_and_gguf
# adapter final is saved at training/adapter_v2/final on Drive
```
(Or rerun with the trainer's built-in export; then create the Ollama Modelfile
from the repo and `ollama create nanohat:360m -f Modelfile`.)

### Cell 6 — EVALUATE
Bring eval_results back for scoring:
```python
!python eval/eval_harness.py --backend ollama --model nanohat:360m \
    --eval dataset/v2/eval.json --limit 300 --out eval_results.json
```
Targets: format compliance >98%, refusal precision >95%, id-threading >90%.

## TIMELINE MATH
~25M train tokens x 3 epochs on T4 (fp16, LoRA r=32) ~= 3-5 GPU hours total.
Free sessions are ~4h: expect 1-2 sessions with --resume between them.
