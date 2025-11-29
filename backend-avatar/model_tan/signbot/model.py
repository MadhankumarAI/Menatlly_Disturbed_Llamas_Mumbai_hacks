# server/model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class KPTransformer(nn.Module):
    """
    Keypoint-based Transformer classifier.
    Input: (B, T, input_dim)
    Output: (B, num_classes) logits
    """
    def __init__(self, input_dim, num_classes, d_model=192, nhead=6, num_layers=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.input_linear(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)   # (B, T, d_model)
        x = x.transpose(1, 2)     # (B, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        logits = self.classifier(x)   # (B, num_classes)
        return logits
