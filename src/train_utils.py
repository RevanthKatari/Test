"""
Training and evaluation utilities for multimodal models.

All models receive (text, image, audio, video) as the first 4 positional
arguments. Text-only models accept and ignore the extra modalities.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for text, image, audio, video, y in loader:
        text  = text.to(device)
        image = image.to(device)
        audio = audio.to(device)
        video = video.to(device)
        y     = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(text, image, audio, video)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return (running_loss / len(loader.dataset),
            accuracy_score(all_labels, all_preds))


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for text, image, audio, video, y in loader:
            text  = text.to(device)
            image = image.to(device)
            audio = audio.to(device)
            video = video.to(device)
            y     = y.to(device)

            logits, _ = model(text, image, audio, video)
            loss = criterion(logits, y)

            running_loss += loss.item() * y.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics = {
        "accuracy":  round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "auc":       round(roc_auc_score(all_labels, all_probs), 4),
    }
    return avg_loss, metrics, np.array(all_preds), np.array(all_probs), np.array(all_labels)


def get_attention_weights(model, loader, device, max_samples=200):
    """Extract attention / gate weights for visualization."""
    model.eval()
    w_list, lbl_list = [], []

    with torch.no_grad():
        collected = 0
        for text, image, audio, video, y in loader:
            text  = text.to(device)
            image = image.to(device)
            audio = audio.to(device)
            video = video.to(device)
            _, attn_w = model(text, image, audio, video)
            w_list.append(attn_w.cpu().numpy())
            lbl_list.extend(y.numpy())
            collected += y.size(0)
            if collected >= max_samples:
                break

    return (np.concatenate(w_list)[:max_samples],
            np.array(lbl_list)[:max_samples])
