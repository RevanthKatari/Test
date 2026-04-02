import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR        = os.path.join(PROJECT_DIR, "data")
EMBEDDINGS_DIR  = os.path.join(DATA_DIR, "embeddings")
RESULTS_DIR     = os.path.join(PROJECT_DIR, "results")
MODELS_DIR      = os.path.join(PROJECT_DIR, "saved_models")
PRETRAINED_DIR  = os.path.join(MODELS_DIR, "pretrained")

DATASET_PATH    = os.path.join(DATA_DIR, "WELFake_Dataset.csv")

# ── Embedding files (pre-computed) ────────────────────────────
TEXT_EMB_PATH   = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy")
IMAGE_EMB_PATH  = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
AUDIO_EMB_PATH  = os.path.join(EMBEDDINGS_DIR, "audio_embeddings.npy")
VIDEO_EMB_PATH  = os.path.join(EMBEDDINGS_DIR, "video_embeddings.npy")
LABELS_PATH     = os.path.join(EMBEDDINGS_DIR, "labels.npy")

# ── Embedding dimensions ──────────────────────────────────────
TEXT_DIM   = 768       # all-mpnet-base-v2
IMAGE_DIM  = 1280      # CLIP (512) + DINOv2 (768)
AUDIO_DIM  = 768       # Wav2Vec2
VIDEO_DIM  = 768       # VideoMAE
HIDDEN_DIM = 512       # projection dimension for all models

# ── Model architecture ────────────────────────────────────────
NUM_CLASSES    = 2
NUM_LAYERS     = 2
DROPOUT        = 0.3
NUM_FILTERS    = 64
KERNEL_SIZES   = [3]     # for CNN across 4 modalities (seq_len=4)

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
NUM_EPOCHS     = 15
PATIENCE       = 5
RANDOM_SEED    = 42

# ── Splits ────────────────────────────────────────────────────
TEST_SIZE = 0.20
VAL_SIZE  = 0.10
