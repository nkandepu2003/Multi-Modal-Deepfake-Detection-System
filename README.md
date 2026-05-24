Deepfake Detection — Multi-Modal Ensemble with Grad-CAM Explainability
98.13% accuracy · 0.9902 AUC-ROC · 100% cross-dataset accuracy on FaceForensics++

What This Project Is About
Deepfakes are AI-generated videos where someone's face is replaced using deep learning. They are increasingly being used for misinformation, fraud, and other harmful purposes — and detecting them reliably is still an open problem.
I built a deepfake detection system that goes beyond just saying "this video is fake." It analyses three different aspects of every video simultaneously and generates a visual heatmap showing exactly which part of the face gave it away. That makes it actually useful for forensic investigation, not just classification.

The Problem I Was Trying to Solve
Most existing deepfake detectors have two major weaknesses.
The first is that they only look at one type of feature — usually pixel-level appearance. This means they perform well on the dataset they were trained on but fail badly when tested on a different one. Before I applied fine-tuning in this project, my model dropped from 99% accuracy on its training dataset down to 56.5% on a completely different one. That gap is a known problem in deepfake detection research.
The second weakness is that they give no explanation. A confidence score of "94% FAKE" tells an investigator nothing about where the manipulation happened or what to look for. For real-world use, explainability is not optional.
This project addresses both.

How I Solved It
Three models, three perspectives
Instead of relying on a single model, I built three independent ones that each look at the video differently:

Spatial model (EfficientNet-B4) looks at the raw pixels and detects visual inconsistencies like unnatural skin texture, blurry edges, and blending artifacts around the face boundary
Spectral model (FFT CNN) converts each frame into a frequency map and detects patterns that GAN-generated faces leave behind — patterns that are invisible to the human eye but very visible to a trained model
Forensic model (Landmark MLP) extracts 68 facial geometry points and detects the subtle structural inconsistencies that face-swapping algorithms introduce

Their predictions are combined using weighted averaging — the spatial model carries 90% of the weight since it is the strongest, with spectral and forensic contributing 5% each. A combined score above 0.60 means FAKE.
Making predictions explainable with Grad-CAM
After every prediction, the system generates a Grad-CAM heatmap on the most suspicious frame. It highlights the regions — usually around the eyes, jawline, or skin boundary — that the model focused on. This turns a black-box classification into something a human investigator can actually use.

Results
MetricTargetResultAccuracy on Celeb-DF v2 test set≥ 90%98.13%AUC-ROC≥ 0.940.9902Accuracy on FaceForensics++ (never seen during training)≥ 85%100%Grad-CAM working correctlyPer prediction9/10 samples
Individual model performance
ModelValidation AccuracySpatial — EfficientNet-B499.12%Spectral — FFT CNN84.14%Forensic — Landmark MLP72.96%Ensemble (test set)98.13%

Key Challenges and How I Dealt With Them
Cross-dataset generalisation
When I first tested on FaceForensics++, accuracy was only 56.5%. The model had learned Celeb-DF specific artifacts rather than general deepfake patterns. I fixed this by fine-tuning on a mixed dataset — 60% Celeb-DF and 40% FaceForensics++ — for five additional epochs at a much lower learning rate. This let the model adapt to new patterns without losing what it already knew. Accuracy went from 56.5% to 100%.
Class imbalance
The training data had 86.4% fake videos and only 13.6% real. Without addressing this, the model would have just predicted fake for everything. I used PyTorch's WeightedRandomSampler to balance every training batch to 50/50, which pushed real video accuracy from poor performance up to 97.15%.

Pipeline Overview
The project ran on an NVIDIA A100 GPU on a university HPC cluster. Here is the order everything ran in:
StepWhat it doesTimeEDAAnalyse all 6,529 videos, check for corruption, measure class distribution~3 minPreprocessingFace detection with MTCNN, crop to 224×224, stratified 70/15/15 split~2.5 hrSpatial trainingTrain EfficientNet-B4 with balanced sampling for 10 epochs~2 hrSpectral trainingTrain FFT CNN on frequency magnitude maps~2-3 hrForensic trainingTrain Landmark MLP on 136-dimensional geometry vectors~45 minEnsembleRun weighted fusion on 9,789 held-out test frames~7 minGrad-CAMGenerate heatmap visualisations on 10 sample images~5 minFine-tuningAdapt spatial model to FaceForensics++ with mixed data~1 hrCross-dataset testEvaluate on 400 FaceForensics++ videos~10 minAPIDeploy full pipeline as a REST API on GPUongoing

The API
The system is deployed as a REST API that accepts a video upload and returns:
json{
  "prediction": "FAKE",
  "confidence": 68.37,
  "spatial_prob": 0.7012,
  "spectral_prob": 0.5891,
  "forensic_prob": 0.4234,
  "frames_analysed": 10,
  "gradcam_image": "base64_encoded_png"
}
Uploaded videos are automatically deleted after inference — no input data is stored.
The API runs on a private university HPC cluster and is not publicly accessible.

Tech Stack
AreaToolsDeep LearningPyTorch 2.7.1, EfficientNet-B4, custom CNN and MLPComputer VisionOpenCV, MTCNN, NumPy FFTExplainabilityGrad-CAM 1.5.5APIFlask 3.1Evaluationscikit-learn — AUC-ROC, stratified splittingInfrastructureNVIDIA A100 GPU, Slurm HPC job scheduler

Project Files
deepfake_project/
├── code/
│   ├── eda.py                # Dataset analysis
│   ├── preprocess.py         # Face detection, cropping, splitting
│   ├── retrain_balanced.py   # Spatial model training
│   ├── spectral_model.py     # Spectral model training
│   ├── forensic_model.py     # Forensic model training
│   ├── ensemble_model.py     # Ensemble fusion and evaluation
│   ├── gradcam_inference.py  # Heatmap generation
│   ├── crossdataset_test.py  # FaceForensics++ evaluation
│   ├── finetune_mixed.py     # Domain-adaptive fine-tuning
│   └── app.py                # REST API

Limitations

Video-level inference can be less reliable than frame-level evaluation for borderline cases near the classification threshold
The forensic module uses Shi-Tomasi corner detection as a landmark proxy — replacing this with proper dlib landmark extraction would improve precision
Like all deepfake detectors, the system has limitations against very new generation techniques not represented in the training data


Datasets

Celeb-DF v2 — Li et al. (2020) — training and evaluation
FaceForensics++ — Rossler et al. (2019) — cross-dataset testing only
