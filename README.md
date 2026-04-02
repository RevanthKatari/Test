# Multimodal Fake News Detection Using Hybrid CNN-BiGRU with Sequential Attention

University of Windsor — Intro to AI

## Team

| Name | Role |
|---|---|
| Revanth Katari | Model architecture & LLM embeddings |
| Kruthika Shantha Murthy | Data preprocessing & multimodal dataset creation |
| Naga Sai Bharath Potla | CNN + BiGRU implementation & hyperparameter tuning |
| Kavya Pagaria | Sequential attention & ablation / interpretability |
| Sai Srinivas Uppara | Experiments, evaluation & result documentation |

## Architecture

```
Text  (768-dim, all-mpnet-base-v2)  ─┐
Image (1280-dim, CLIP + DINOv2)     ─┤  Project to 512-dim
Audio (768-dim, Wav2Vec2)           ─┤  ──▶ Stack as 4-step sequence
Video (768-dim, VideoMAE)           ─┘
        │
        ▼
    CNN (Conv1d, kernel=3)  ──▶  local cross-modal patterns
        │
        ▼
    BiGRU (2-layer, bidirectional)  ──▶  sequential modeling
        │
        ▼
    Sequential Attention  ──▶  modality weighting + interpretability
        │
        ▼
    Fully Connected  ──▶  Real / Fake
```

## Dataset

**Balanced Multimodal WELFake** — 43,131 samples combining:
- WELFake text-only articles (20%)
- FakeNewsNet text + image articles (30%)
- FakeAVCeleb audio + video samples (50%)

Pre-computed embeddings are stored in `data/embeddings/`.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users**: Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### 2. Run notebooks in order

| # | Notebook | What it does | Time |
|---|---|---|---|
| 1 | `1_data_preparation.ipynb` | EDA + verify pre-computed embeddings | Instant |
| 2 | `2_baseline_models.ipynb` | Train BiLSTM, Gated Fusion; evaluate pretrained | ~10 min (GPU) |
| 3 | `3_hybrid_model.ipynb` | Train proposed CNN-BiGRU-Attention | ~10 min (GPU) |
| 4 | `4_attention_ablation.ipynb` | Ablation study + attention visualization | ~30 min (GPU) |
| 5 | `5_experiment_results.ipynb` | Consolidated comparison & charts | Instant |

### 3. Launch the web app

```bash
cd app
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Folder Structure

```
FINAL/
├── data/
│   ├── WELFake_Dataset.csv              # original 72K articles
│   └── embeddings/                      # pre-computed (43,131 samples)
│       ├── text_embeddings.npy          # (43131, 768)
│       ├── image_embeddings.npy         # (43131, 1280)
│       ├── audio_embeddings.npy         # (43131, 768)
│       ├── video_embeddings.npy         # (43131, 768)
│       └── labels.npy                   # (43131,)
├── src/
│   ├── config.py
│   ├── models.py                        # BiLSTM, GatedFusion, CrossAttn, HybridCNNBiGRU
│   ├── data_utils.py
│   └── train_utils.py
├── notebooks/                           # run in order 1→5
├── app/                                 # Flask web UI
├── saved_models/
│   └── pretrained/                      # teammate's pretrained weights
├── results/                             # auto-generated JSON metrics
├── requirements.txt
└── README.md
```

## Models

| Model | Type | Modalities | Input |
|---|---|---|---|
| BiLSTM + Attention | Baseline | Text only | 768-dim |
| Gated Fusion | Baseline | Text + Image + Audio + Video | 768/1280/768/768 |
| Cross-Attention | Comparison | Text + Image + Audio + Video | 768/1280/768/768 |
| **CNN-BiGRU-Attention** | **Proposed** | Text + Image + Audio + Video | 768/1280/768/768 |

## Portability

Copy the entire `FINAL/` folder, run `pip install -r requirements.txt`, then execute notebooks 1-5.
All paths are relative. Pre-computed embeddings are included.
