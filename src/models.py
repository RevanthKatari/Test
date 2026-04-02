"""
Model architectures for Multimodal Fake News Detection.

Embedding dimensions (pre-computed):
    text  = 768  (all-mpnet-base-v2)
    image = 1280 (CLIP + DINOv2)
    audio = 768  (Wav2Vec2)
    video = 768  (VideoMAE)

Baselines:
  1. BiLSTMTextOnly     — text embeddings + BiLSTM + Attention
  2. GatedFusionModel   — 4-modal gated fusion

Proposed:
  3. HybridCNNBiGRUMultimodal — 4-modal CNN + BiGRU + Sequential Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Sequential Attention
# ------------------------------------------------------------------
class SequentialAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        scores = self.v(torch.tanh(self.W(x)))        # (B, S, 1)
        weights = torch.softmax(scores, dim=1)          # (B, S, 1)
        context = (weights * x).sum(dim=1)              # (B, H)
        return context, weights.squeeze(-1)              # (B, H), (B, S)


# ------------------------------------------------------------------
# Baseline 1 — BiLSTM + Attention (text only)
# ------------------------------------------------------------------
class BiLSTMTextOnly(nn.Module):
    def __init__(self, text_dim=768, hidden=512, seq_len=4,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.chunk = hidden
        self.proj = nn.Linear(text_dim, hidden * seq_len)
        self.bn   = nn.BatchNorm1d(hidden * seq_len)
        self.lstm = nn.LSTM(
            hidden, hidden // 2, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout,
        )
        self.attention = SequentialAttention(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text, image=None, audio=None, video=None):
        x = self.bn(self.proj(text))                   # (B, hidden*seq_len)
        x = x.view(-1, self.seq_len, self.chunk)       # (B, seq_len, hidden)
        out, _ = self.lstm(x)                           # (B, seq_len, hidden)
        ctx, attn_w = self.attention(out)
        return self.head(self.dropout(ctx)), attn_w


# ------------------------------------------------------------------
# Baseline 2 — Gated Fusion (4-modal)
# ------------------------------------------------------------------
class GatedFusionModel(nn.Module):
    """Matches the pretrained gated_fusion_model.pth architecture exactly."""

    def __init__(self, text_dim=768, image_dim=1280, audio_dim=768,
                 video_dim=768, hidden=512, num_classes=2, dropout=0.3):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.video_proj = nn.Linear(video_dim, hidden)

        self.text_gate  = nn.Linear(hidden, hidden)
        self.image_gate = nn.Linear(hidden, hidden)
        self.audio_gate = nn.Linear(hidden, hidden)
        self.video_gate = nn.Linear(hidden, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text, image, audio, video):
        t = self.text_proj(text)
        i = self.image_proj(image)
        a = self.audio_proj(audio)
        v = self.video_proj(video)

        gt = torch.sigmoid(self.text_gate(t))
        gi = torch.sigmoid(self.image_gate(i))
        ga = torch.sigmoid(self.audio_gate(a))
        gv = torch.sigmoid(self.video_gate(v))

        fused = (gt * t + gi * i + ga * a + gv * v) / 4
        gate_weights = torch.stack([gt.mean(-1), gi.mean(-1),
                                    ga.mean(-1), gv.mean(-1)], dim=1)
        return self.classifier(fused), gate_weights   # (B, 2), (B, 4)


# ------------------------------------------------------------------
# Cross-Attention (4-modal) — matches pretrained model
# ------------------------------------------------------------------
class CrossAttentionModel(nn.Module):
    """Matches the pretrained cross_attention_model.pth architecture exactly."""

    def __init__(self, text_dim=768, image_dim=1280, audio_dim=768,
                 video_dim=768, hidden=512, num_classes=2, dropout=0.3):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.video_proj = nn.Linear(video_dim, hidden)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text, image, audio, video):
        t = self.text_proj(text).unsqueeze(1)    # (B, 1, H)
        i = self.image_proj(image).unsqueeze(1)
        a = self.audio_proj(audio).unsqueeze(1)
        v = self.video_proj(video).unsqueeze(1)

        kv = torch.cat([i, a, v], dim=1)          # (B, 3, H)
        fused, attn_w = self.cross_attn(query=t, key=kv, value=kv)
        fused = fused.squeeze(1)                   # (B, H)

        return self.classifier(fused), attn_w.squeeze(1)  # (B, 2), (B, 3)


# ------------------------------------------------------------------
# Proposed — Hybrid CNN + BiGRU + Sequential Attention (4-modal)
# ------------------------------------------------------------------
class HybridCNNBiGRUMultimodal(nn.Module):
    """
    Projects each modality to hidden dim, stacks as a 4-step sequence,
    then applies CNN + BiGRU + Sequential Attention.
    Supports ablation via use_cnn / use_attention flags.
    """

    def __init__(self, text_dim=768, image_dim=1280, audio_dim=768,
                 video_dim=768, hidden=512, num_filters=64,
                 num_classes=2, dropout=0.3,
                 use_cnn=True, use_attention=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_attention = use_attention

        self.text_proj  = nn.Linear(text_dim, hidden)
        self.image_proj = nn.Linear(image_dim, hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.video_proj = nn.Linear(video_dim, hidden)
        self.proj_norm  = nn.LayerNorm(hidden)

        if use_cnn:
            self.conv = nn.Sequential(
                nn.Conv1d(hidden, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            gru_input = num_filters
        else:
            gru_input = hidden

        self.gru = nn.GRU(
            gru_input, hidden // 2, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout,
        )

        if use_attention:
            self.attention = SequentialAttention(hidden)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text, image, audio, video):
        t = F.relu(self.text_proj(text))     # (B, H)
        i = F.relu(self.image_proj(image))
        a = F.relu(self.audio_proj(audio))
        v = F.relu(self.video_proj(video))

        # stack as sequence: (B, 4, H)
        x = torch.stack([t, i, a, v], dim=1)
        x = self.proj_norm(x)

        if self.use_cnn:
            x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 4, F)

        gru_out, _ = self.gru(x)             # (B, 4, H)

        if self.use_attention:
            ctx, attn_w = self.attention(gru_out)
        else:
            ctx = gru_out.mean(dim=1)
            attn_w = torch.ones(gru_out.size(0), 4,
                                device=gru_out.device) / 4

        return self.head(self.dropout(ctx)), attn_w
