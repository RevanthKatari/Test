"""
Flask app for Multimodal Fake News Detection.

Inference uses text embedding (all-mpnet-base-v2) with the trained
Hybrid CNN-BiGRU-Attention model. Image/audio/video are zero-padded
for text-only web input.

Dashboard shows all model metrics from results/ JSONs.
"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify

from config import (
    TEXT_DIM, IMAGE_DIM, AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM,
    NUM_CLASSES, DROPOUT, NUM_FILTERS, MODELS_DIR, RESULTS_DIR,
)
from models import HybridCNNBiGRUMultimodal

app = Flask(__name__)

_st_model = None
_model = None
_device = None
_all_metrics = {}


def _load():
    global _st_model, _model, _device, _all_metrics

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-mpnet-base-v2")
    print("[app] Loaded text embedding model (all-mpnet-base-v2)")

    weight_path = os.path.join(MODELS_DIR, "hybrid_cnn_bigru_attn.pt")
    if os.path.exists(weight_path):
        _model = HybridCNNBiGRUMultimodal(
            text_dim=TEXT_DIM, image_dim=IMAGE_DIM,
            audio_dim=AUDIO_DIM, video_dim=VIDEO_DIM,
            hidden=HIDDEN_DIM, num_filters=NUM_FILTERS,
            num_classes=NUM_CLASSES, dropout=DROPOUT,
        ).to(_device)
        _model.load_state_dict(
            torch.load(weight_path, map_location=_device, weights_only=True))
        _model.eval()
        print(f"[app] Loaded hybrid model from {weight_path}")
    else:
        print(f"[app] WARNING: no model at {weight_path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for fp in sorted(os.listdir(RESULTS_DIR)):
        if fp.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, fp)) as f:
                data = json.load(f)
            name = fp.replace("_metrics.json", "").replace("_", " ").title()
            _all_metrics[name] = data.get("test", data)


def _predict(text):
    if _model is None or _st_model is None:
        return {"error": "Model not loaded. Train it first (notebook 3)."}

    text_emb = _st_model.encode([text])[0].astype(np.float32)

    t = torch.tensor(text_emb).unsqueeze(0).to(_device)
    i = torch.zeros(1, IMAGE_DIM).to(_device)
    a = torch.zeros(1, AUDIO_DIM).to(_device)
    v = torch.zeros(1, VIDEO_DIM).to(_device)

    with torch.no_grad():
        logits, attn_w = _model(t, i, a, v)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(logits.argmax(1).item())
        weights = attn_w[0].cpu().numpy()

    modalities = ["Text", "Image", "Audio", "Video"]
    modality_attention = [
        {"modality": m, "weight": round(float(w), 4)}
        for m, w in zip(modalities, weights)
    ]

    return {
        "label": "FAKE" if pred == 1 else "REAL",
        "confidence": round(float(probs.max()), 4),
        "prob_real": round(float(probs[0]), 4),
        "prob_fake": round(float(probs[1]), 4),
        "modality_attention": modality_attention,
    }


@app.route("/")
def index():
    return render_template("index.html", metrics=_all_metrics)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(_predict(text))


if __name__ == "__main__":
    _load()
    print("[app] http://localhost:5000")
    app.run(debug=False, port=5000)
