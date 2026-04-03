import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import gradio as gr
from torchvision.models import efficientnet_b2

# ──────────────────────────────────────────────
# Constants (must match training)
# ──────────────────────────────────────────────
SR = 22050
DURATION = 15
SAMPLES = SR * DURATION
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

# ──────────────────────────────────────────────
# Model Definition (same as training notebook)
# ──────────────────────────────────────────────
class AudioEfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        backbone = efficientnet_b2(weights=None)

        # Adapt first conv: RGB → 1 channel
        weight = backbone.features[0][0].weight.data
        backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        backbone.features[0][0].weight.data = weight.mean(dim=1, keepdim=True)

        # Replace classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────
# Load model from checkpoint
# ──────────────────────────────────────────────
def load_model(ckpt_path="model.ckpt"):
    """Load a PyTorch-Lightning checkpoint into a plain nn.Module."""
    model = AudioEfficientNet(num_classes=len(GENRES))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Lightning saves weights under "state_dict" with a "model." prefix
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        # Remove "model.model." → "model."  (Lightning wrapper)
        new_key = k.replace("model.model.", "model.")
        # Also handle if key starts with just "model."
        if new_key.startswith("model."):
            cleaned[new_key] = v
        else:
            cleaned["model." + new_key] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


MODEL = load_model("model.ckpt")

# ──────────────────────────────────────────────
# Audio preprocessing (same as training)
# ──────────────────────────────────────────────
def preprocess_audio(audio_path):
    """Load audio, clip/pad to fixed duration, convert to mel spectrogram."""
    audio, sr = librosa.load(audio_path, sr=SR, mono=True)

    # Clip or pad to fixed length
    if len(audio) > SAMPLES:
        start = (len(audio) - SAMPLES) // 2     # center crop for inference
        audio = audio[start : start + SAMPLES]
    else:
        pad = SAMPLES - len(audio)
        audio = np.pad(audio, (0, pad))

    # Mel spectrogram → log scale
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel)

    # Shape: (1, 1, n_mels, time)
    tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).float()
    return tensor


# ──────────────────────────────────────────────
# Prediction function
# ──────────────────────────────────────────────
def predict_genre(audio_path):
    """Run inference and return genre probabilities."""
    if audio_path is None:
        return {g: 0.0 for g in GENRES}

    mel = preprocess_audio(audio_path)

    with torch.no_grad():
        logits = MODEL(mel)
        probs = torch.softmax(logits, dim=1).squeeze()

    return {genre: float(prob) for genre, prob in zip(GENRES, probs)}


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath", label="Upload an audio file"),
    outputs=gr.Label(num_top_classes=5, label="Genre Prediction"),
    title="Audio Genre Classifier",
    description=(
        "Upload an audio file and the model will classify it into one of 10 genres: "
        "blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, or rock.\n\n"
        "**Model:** Fine-tuned EfficientNet-B2 trained on synthetic mashups with mel-spectrogram features."
    ),
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())