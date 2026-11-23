"""CRNN handwriting recogniser (report 5.3.2, Figure 4).

Six convolution layers with batch normalisation and max pooling turn a 64 x W
grayscale line into W/4 feature vectors; two bidirectional LSTM layers model
context in both directions (which lets the network emit the short i-matra after
its consonant although it is written before it); a linear layer gives a
distribution over the charset plus the CTC blank. About 8 M parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .charset import NUM_CLASSES, decode_ids


def _conv(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, lstm_units: int = 256, dropout_cnn: float = 0.2,
                 dropout_lstm: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv(1, 64), nn.MaxPool2d(2, 2),                       # 32 x W/2
            *_conv(64, 128), nn.MaxPool2d(2, 2),                     # 16 x W/4
            *_conv(128, 256), *_conv(256, 256), nn.MaxPool2d((2, 1)),  # 8 x W/4
            nn.Dropout2d(dropout_cnn),
            *_conv(256, 512), *_conv(512, 512), nn.MaxPool2d((2, 1)),  # 4 x W/4
            nn.Dropout2d(dropout_cnn),
        )
        self.proj = nn.Linear(512 * 4, 256)
        self.rnn = nn.LSTM(256, lstm_units, num_layers=2, bidirectional=True, batch_first=True,
                           dropout=dropout_lstm)
        self.drop = nn.Dropout(dropout_lstm)
        self.head = nn.Linear(2 * lstm_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B x 1 x 64 x W (ink bright, values in [0, 1]) -> log-probs T x B x C with T = W / 4."""
        f = self.cnn(x)                                   # B x 512 x 4 x T
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)    # B x T x 2048
        f, _ = self.rnn(torch.relu(self.proj(f)))
        return self.head(self.drop(f)).log_softmax(-1).permute(1, 0, 2)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor | None = None) -> list[tuple[str, float]]:
    """Best-path decoding. Returns (text, raw confidence) per item.

    Raw confidence = geometric mean of the posteriors of the emitted characters (report 5.3.8);
    an empty output gets the mean blank posterior instead.
    """
    lp = log_probs.detach().cpu()
    best, idx = lp.max(-1)                                # T x B
    out = []
    for b in range(lp.shape[1]):
        t_max = int(lengths[b]) if lengths is not None else lp.shape[0]
        ids, logs, prev = [], [], 0
        for t in range(t_max):
            i = int(idx[t, b])
            if i != 0 and i != prev:
                ids.append(i)
                logs.append(float(best[t, b]))
            prev = i
        conf = math.exp(sum(logs) / len(logs)) if logs else math.exp(float(best[:t_max, b].mean()))
        out.append((decode_ids(ids), conf))
    return out
