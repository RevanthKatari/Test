"""
Data loading for pre-computed multimodal embeddings.

Files expected in data/embeddings/:
    text_embeddings.npy   (N, 768)
    image_embeddings.npy  (N, 1280)
    audio_embeddings.npy  (N, 768)
    video_embeddings.npy  (N, 768)
    labels.npy            (N,)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MultimodalDataset(Dataset):
    def __init__(self, text, image, audio, video, labels):
        self.text   = torch.tensor(text, dtype=torch.float32)
        self.image  = torch.tensor(image, dtype=torch.float32)
        self.audio  = torch.tensor(audio, dtype=torch.float32)
        self.video  = torch.tensor(video, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx],
                self.audio[idx], self.video[idx],
                self.labels[idx])


def load_embeddings(emb_dir):
    """Load all pre-computed embeddings and labels."""
    text   = np.load(os.path.join(emb_dir, "text_embeddings.npy")).astype(np.float32)
    image  = np.load(os.path.join(emb_dir, "image_embeddings.npy")).astype(np.float32)
    audio  = np.load(os.path.join(emb_dir, "audio_embeddings.npy")).astype(np.float32)
    video  = np.load(os.path.join(emb_dir, "video_embeddings.npy")).astype(np.float32)
    labels = np.load(os.path.join(emb_dir, "labels.npy")).astype(np.int64)

    print(f"[data] Loaded {len(labels):,} samples")
    print(f"       text={text.shape} image={image.shape} "
          f"audio={audio.shape} video={video.shape}")
    return text, image, audio, video, labels


def create_splits(text, image, audio, video, labels,
                  test_size=0.20, val_size=0.10, seed=42):
    """Stratified train / val / test split across all modalities."""
    idx = np.arange(len(labels))
    idx_tmp, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels)
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=val_frac, random_state=seed, stratify=labels[idx_tmp])

    def _slice(indices):
        return (text[indices], image[indices],
                audio[indices], video[indices], labels[indices])

    return _slice(idx_train), _slice(idx_val), _slice(idx_test)


def make_dataloader(split_tuple, batch_size=32, shuffle=True):
    """Create DataLoader from a (text, image, audio, video, labels) tuple."""
    ds = MultimodalDataset(*split_tuple)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
