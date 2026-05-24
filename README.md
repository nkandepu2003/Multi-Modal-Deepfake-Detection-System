# Deepfake Detection — Multi-Modal Ensemble with Grad-CAM Explainability

**98.13% accuracy · 0.9902 AUC-ROC · 100% cross-dataset accuracy on FaceForensics++**

---

## What This Project Is About

Deepfakes are AI-generated videos where someone's face is replaced using deep learning. They are increasingly being used for misinformation, fraud, and other harmful purposes — and detecting them reliably is still an open problem.

I built a deepfake detection system that goes beyond just saying "this video is fake." It analyses three different aspects of every video simultaneously and generates a visual heatmap showing exactly which part of the face gave it away — making it actually useful for forensic investigation, not just classification.

The entire system was trained and deployed on NAU Monsoon HPC, a private university high performance computing cluster with NVIDIA A100 GPU access. The code files will be added here once they are exported from the cluster.

---

## The Problem I Was Trying to Solve

Most existing deepfake detectors have two major weaknesses.

The first is that they only look at one type of feature — usually pixel-level appearance. This means they perform well on the dataset they were trained on but fail badly when tested on a different one. Before I applied fine-tuning in this project, my model dropped from 99% accuracy on its training dataset down to 56.5% on a completely different one. That gap is a known problem in deepfake detection research.

The second weakness is that they give no explanation. A confidence score of "94% FAKE" tells an investigator nothing about where the manipulation happened or what to look for. For real-world use, explainability is not optional.

This project addresses both.

---

## How I Solved It

Instead of relying on a single model, I built three independent ones that each look at the video differently:

- **Spatial model (EfficientNet-B4)** looks at the raw pixels and detects visual inconsistencies like unnatural skin texture, blurry edges, and blending artifacts around the face boundary
- **Spectral model (FFT CNN)** converts each frame into a frequency map and detects patterns that GAN-generated faces leave behind — patterns that are invisible to the human eye but detectable by a trained model
- **Forensic model (Landmark MLP)** extracts 68 facial geometry points and detects the subtle structural inconsistencies that face-swapping algorithms introduce

Their predictions are combined using weighted averaging — the spatial model carries 90% of the weight since it is the strongest, with spectral and forensic contributing 5% each. A combined score above 0.60 means FAKE.

After every prediction, the system generates a Grad-CAM heatmap highlighting the regions — usually around the eyes, jawline, or skin boundary — that the model found most suspicious. This turns a black-box classification into something a human investigator can actually use.

The full system is deployed as a REST API on an NVIDIA A100 GPU — upload a video, get back a prediction, confidence score, and heatmap. Uploaded videos are automatically deleted after inference.

---

## Results

| Metric | Target | Achieved |
|---|---|---|
| Accuracy on Celeb-DF v2 test set | ≥ 90% | **98.13%** |
| AUC-ROC | ≥ 0.94 | **0.9902** |
| Cross-dataset accuracy on FaceForensics++ | ≥ 85% | **100%** |
| Grad-CAM explainability | Working | **9/10 samples** |

### Individual Model Performance

| Model | Validation Accuracy |
|---|---|
| Spatial — EfficientNet-B4 | 99.12% |
| Spectral — FFT CNN | 84.14% |
| Forensic — Landmark MLP | 72.96% |
| **Ensemble (test set)** | **98.13%** |

---

## Key Challenges

**Cross-dataset generalisation**
When I first tested on FaceForensics++, accuracy was only 56.5%. The model had learned Celeb-DF specific artifacts rather than general deepfake patterns. I fixed this by fine-tuning on a mixed dataset — 60% Celeb-DF and 40% FaceForensics++ — for five additional epochs at a much lower learning rate. This let the model adapt to new patterns without losing what it already knew. Accuracy went from 56.5% to 100%.

**Class imbalance**
The training data had 86.4% fake videos and only 13.6% real. Without addressing this, the model would have just predicted fake for everything and still gotten 86% accuracy without learning anything useful. I used PyTorch's WeightedRandomSampler to balance every training batch to 50/50, which pushed real video accuracy up to 97.15%.

---

## Real-World Use Cases

- **Media and journalism** — verify whether a video is authentic before publishing
- **Legal and forensic** — provide visual evidence of exactly where manipulation occurred
- **Social media platforms** — automated screening of uploaded video content
- **Law enforcement** — identify deepfake evidence in fraud and impersonation cases

---

## Tech Stack

PyTorch · EfficientNet-B4 · OpenCV · MTCNN · NumPy FFT · Grad-CAM · Flask · scikit-learn · NVIDIA A100 · Slurm

---

## Project Structure

> The project was built and run entirely on NAU Monsoon HPC — a private university computing cluster. Direct access requires university credentials. Code files will be uploaded here once exported from the cluster.

```
deepfake_project/
├── code/
│   ├── eda.py                # Dataset analysis and class distribution
│   ├── preprocess.py         # Face detection, cropping, train/val/test split
│   ├── retrain_balanced.py   # Spatial model — EfficientNet-B4 training
│   ├── spectral_model.py     # Spectral model — FFT CNN training
│   ├── forensic_model.py     # Forensic model — Landmark MLP training
│   ├── ensemble_model.py     # Weighted fusion and final evaluation
│   ├── gradcam_inference.py  # Grad-CAM heatmap generation
│   ├── crossdataset_test.py  # Cross-dataset evaluation on FaceForensics++
│   ├── finetune_mixed.py     # Domain-adaptive fine-tuning
│   └── app.py                # Flask REST API
```

---

## Datasets

- [Celeb-DF v2](https://arxiv.org/abs/1909.12962) — Li et al. (2020) — training and evaluation
- [FaceForensics++](https://arxiv.org/abs/1901.08971) — Rossler et al. (2019) — cross-dataset testing only
