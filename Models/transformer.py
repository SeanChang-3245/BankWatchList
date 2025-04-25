import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        feat_dim,       # ← number of features per time‐step
        d_model=128,    # you can shrink this from 512
        nhead=4,
        num_layers=2,
        num_classes=1,  # if you're doing binary (watchlist yes/no)
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        # instead of nn.Embedding, use a linear projector
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        encoder_layer    = nn.TransformerEncoderLayer(d_model, nhead,
                                                      dim_feedforward,
                                                      dropout)
        self.encoder     = nn.TransformerEncoder(encoder_layer,
                                                 num_layers)
        self.classifier  = nn.Linear(d_model, num_classes)

    def forward(self, seq, src_mask=None, src_key_padding_mask=None):
        # seq: (batch, seq_len, feat_dim)
        x = self.input_proj(seq)       # → (batch, seq_len, d_model)
        x = self.pos_enc(x)            # add positional
        x = x.transpose(0,1)           # → (seq_len, batch, d_model)
        out = self.encoder(x,
                           mask=src_mask,
                           src_key_padding_mask=src_key_padding_mask)
        out = out.mean(dim=0)          # (batch, d_model)
        return self.classifier(out)    # (batch, num_classes)