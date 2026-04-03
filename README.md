# Audio Genre Classification with Deep Learning

**Student:** Miriyala Indu Vardhan | **ID:** 21F3000478  
**Course:** Deep Learning (IIT Madras)

A complete pipeline for automatically classifying music into 10 genres using synthetic mashup data and transfer learning. The project progresses through 5 milestones — from exploratory analysis to a live Gradio web app deployed on Hugging Face Spaces.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Genres](#genres)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [Model Architectures](#model-architectures)
- [Training Pipeline](#training-pipeline)
- [Deployment](#deployment)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)

---

## Project Overview

The core challenge is classifying audio recordings into 10 music genres. Rather than using raw audio clips, the approach generates **synthetic mashups** from isolated instrument stems (drums, bass, vocals, other), mixing them with environmental noise and audio augmentations to create a rich training set that generalizes well.

Three neural network architectures are explored and compared:
- A custom CNN built from scratch
- Fine-tuned ResNet-18
- Fine-tuned EfficientNet-B2 *(best performing, used in deployment)*

All models treat audio classification as an image classification problem by converting waveforms into **mel spectrograms** (128 × N time-frequency images).

---

## Genres

The classifier recognizes 10 genres:

| # | Genre | # | Genre |
|---|-------|---|-------|
| 0 | Blues | 5 | Jazz |
| 1 | Classical | 6 | Metal |
| 2 | Country | 7 | Pop |
| 3 | Disco | 8 | Reggae |
| 4 | Hip-Hop | 9 | Rock |

---

## Dataset

### Source Data

| Dataset | Description |
|---------|-------------|
| **Genre Stems** (`messy_mashup/genres_stems/`) | Isolated instrument stems (drums, bass, vocals, other) for each song across 10 genres |
| **ESC-50 Noise** (`ESC-50-master/audio/`) | Environmental sound recordings used as background noise during augmentation |
| **Test Set** | CSV with `id` and `filename` columns for final predictions |

### Synthetic Data Generation

Because the stems dataset alone is small, `MashupDataset` generates ~**7,000 synthetic training samples** (700 per genre) by:

1. Sampling 4 random stems from a single genre
2. Applying random **time-stretching** (via PyRubberband) to each stem
3. Mixing stems with random gain weights (0.5–1.5)
4. Adding random noise tracks from ESC-50 at a target SNR of 10 dB
5. Applying random resampling
6. Converting the mixed waveform to a **128-band mel spectrogram**

### Audio Processing Constants

```python
SR          = 22050   # Sampling rate (Hz)
DURATION    = 5.0     # Clip length (seconds)
N_FFT       = 2048    # FFT window size
HOP_LENGTH  = 512     # Hop size between STFT frames
N_MELS      = 128     # Number of mel filter banks
TARGET_SNR  = 10 dB   # Noise mixing target
```

---

## Project Structure

```
.
├── 21f3000478_kaggle_notebook.ipynb     # Main training notebook (Kaggle)
├── MileStone_1_and_2.ipynb              # EDA & data quality analysis
├── MileStone_3.ipynb                    # NN fundamentals exercises
├── MileStone_4.ipynb                    # Synthetic data gen & CRNN
├── MileStone_5.ipynb                    # Audio Spectrum Transformer (AST)
├── 21F3000478_Miriyala_Indu_Vardhan.pdf # Full project report
├── requirements.txt                     # Training dependencies
└── Deployment code/
    ├── app.py                           # Gradio web application
    ├── model.ckpt                       # Trained EfficientNet-B2 checkpoint
    └── requirements.txt                 # Deployment dependencies
```

---

## Milestones

### Milestone 1 & 2 — Exploratory Data Analysis (`MileStone_1_and_2.ipynb`)

- Loads and navigates the genre stems hierarchy
- Builds train/validation splits
- Detects **corrupted files** (< 4 KB) and **abnormally large files** (> 5 MB)
- Identifies **long silences** (using top-dB thresholding)
- Computes per-genre and per-stem file size statistics

### Milestone 3 — Neural Network Fundamentals (`MileStone_3.ipynb`)

Hands-on exercises covering:
- Tensor reshaping, flattening, and batch operations
- Mel spectrogram transformation from raw waveforms
- Convolution and pooling layer mechanics
- Linear layer parameter counting
- Cross-entropy loss computation
- Manual gradient descent and weight update steps
- Weights & Biases (W&B) logging setup

### Milestone 4 — Synthetic Data & CRNN (`MileStone_4.ipynb`)

- Implements the `PrecomputedFeatureDataset` class for cached features
- Defines a **CRNN** (Convolutional Recurrent Neural Network) model
- Counts LSTM layer parameters analytically
- Sets up DataLoaders with reproducible seeds

### Milestone 5 — Audio Spectrum Transformer (`MileStone_5.ipynb`)

- Loads the Hugging Face `ASTFeatureExtractor` and `ASTForAudioClassification`
- Adapts the AST model head for 10-class genre classification
- Counts total vs. trainable parameters
- Analyzes audio normalization requirements for transformer inputs

### Main Kaggle Notebook — Full Training Pipeline (`21f3000478_kaggle_notebook.ipynb`)

Complete end-to-end workflow:
1. Data loading and preprocessing
2. Synthetic mashup generation (700 samples/genre)
3. DataLoader setup with fixed seeds for reproducibility
4. Training all three model architectures with PyTorch Lightning
5. Model uploading to Kaggle Hub
6. Test set inference and prediction aggregation
7. Submission CSV generation

---

## Model Architectures

### 1. AudioCNN (Custom)

A CNN built from scratch with three convolutional blocks:

```
Conv2d(1→16) → BN → ReLU → MaxPool
Conv2d(16→32) → BN → ReLU → MaxPool
Conv2d(32→64) → BN → ReLU → AdaptiveAvgPool(4×4)
Flatten → Linear(1024→128) → ReLU → Dropout(0.5)
Linear(128→10)
```

### 2. AudioResNet18

Pretrained **ResNet-18** adapted for 1-channel grayscale mel spectrograms:
- First conv layer modified: `Conv2d(3→1, kernel=7, stride=2, padding=3)`
- Final FC layer replaced: `Linear(512→10)`

### 3. AudioEfficientNet (Best)

Pretrained **EfficientNet-B2** adapted for audio:
- First conv layer modified: `Conv2d(3→1, ...)` (grayscale input)
- Classifier head replaced: `Linear(→10)`
- Trained with learning rate `5e-4`

All models are wrapped in a **PyTorch Lightning module** (`AudioLightningModule`) that handles:
- Forward pass
- Training and validation steps
- Accuracy and weighted F1-score logging
- Best checkpoint saving by `val_f1`

---

## Training Pipeline

### Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Max epochs | 50 |
| Optimizer | Adam |
| LR (CNN) | 1e-3 |
| LR (ResNet-18) | 1e-3 |
| LR (EfficientNet-B2) | 5e-4 |
| Precision | 16-bit mixed |
| Metrics | Accuracy, Weighted F1-Score |
| Checkpoint | Best `val_f1` |
| Random seeds | Data: 67, Training: 1234 |

### Experiment Tracking

Training runs are logged to **Weights & Biases (W&B)**. Trained models are versioned and stored on **Kaggle Hub**.

---

## Deployment

The best model (EfficientNet-B2) is deployed as a **Gradio web application** on Hugging Face Spaces.

### How It Works (`Deployment code/app.py`)

1. **Load model**: Reads `model.ckpt` (PyTorch Lightning checkpoint), remaps state dict keys for standalone inference
2. **Preprocess audio**: Load with Librosa → normalize → trim silence → compute 128-band mel spectrogram → resize to `(1, 128, 216)`
3. **Predict**: Run mel spectrogram through EfficientNet-B2, apply softmax
4. **Return**: Dictionary of genre → probability for all 10 classes

### Gradio Interface

- **Input**: Upload an audio file (any common format)
- **Output**: Bar chart of confidence scores for all 10 genres
- **SDK**: Gradio 6.11.0, Soft theme

### Run Locally

```bash
cd "Deployment code"
pip install -r requirements.txt
python app.py
```

---

## Results

The final model is **EfficientNet-B2** fine-tuned on 7,000 synthetic mashup samples. It is evaluated on a held-out validation split using **weighted F1-score** as the primary metric (to account for any class imbalance).

For full quantitative results, architecture comparisons, and analysis, see the project report:  
`21F3000478_Miriyala_Indu_Vardhan.pdf`

---

## Setup & Installation

### Training Environment (Kaggle)

The main notebook is designed to run in a Kaggle notebook with GPU acceleration. Dependencies are pre-installed in that environment.

### Local Setup

```bash
# Clone the repository
git clone <repo-url>
cd DL-21f3000478-notebook-t12026-main

# Install dependencies
pip install -r requirements.txt
```

**Root `requirements.txt`:**
```
torch
torchvision
pytorch-lightning
torchmetrics
librosa
numpy
pandas
pyrubberband
wandb
kagglehub
```

**Deployment `requirements.txt`:**
```
torch
torchvision
librosa
numpy
gradio
```

---

## Usage

### Run the Web App

```bash
cd "Deployment code"
python app.py
```

Open the local Gradio URL in your browser, upload any audio file, and the app will output genre probabilities.

### Run Training (Kaggle)

Open `21f3000478_kaggle_notebook.ipynb` in a Kaggle notebook environment with:
- GPU accelerator enabled
- The `messy_mashup` and `ESC-50-master` datasets attached

Run all cells sequentially. The trained checkpoint will be saved and uploaded to Kaggle Hub.

### Explore Milestones Locally

```bash
jupyter notebook MileStone_1_and_2.ipynb
jupyter notebook MileStone_3.ipynb
jupyter notebook MileStone_4.ipynb
jupyter notebook MileStone_5.ipynb
```

> Note: Milestone notebooks reference Kaggle dataset paths. Update paths to local equivalents if running outside Kaggle.

