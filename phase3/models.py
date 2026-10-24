"""
models.py
---------
Model definitions for Phase 3.

Contains:
  - ConvLSTMCell / ConvLSTM: the main model (spatial + temporal), following
    the same design idea as the base paper (Shi et al. 2015 ConvLSTM,
    stacked layers -> Conv2D classifier head -> sigmoid).
  - SimpleLSTM: the "plain LSTM" baseline from the paper -- handles time,
    but (unlike ConvLSTM) has no convolution, so it looks at each grid
    cell's own history independently and ignores its neighbors.

Beginner note: think of ConvLSTM as "LSTM cells where the math inside
each gate is a small image-convolution instead of a plain matrix
multiply" -- that's what lets it learn spatial patterns (like "crime
often appears next to last week's crime") on top of temporal ones.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM layer's cell. At every time step it takes the current
    input frame + the previous hidden/cell state, and produces a new
    hidden/cell state -- using convolutions instead of the fully-connected
    matrix multiplications a normal LSTM cell would use.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2  # "same" padding, keeps H x W unchanged
        # One conv produces all 4 gates at once (input, forget, output,
        # candidate cell) -- a common efficiency trick, stacking them along
        # the channel dimension and splitting afterward.
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)  # stack along channel dim
        combined_conv = self.conv(combined)

        i, f, o, g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate cell content

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """
    Stacks several ConvLSTMCell layers, then a Conv2D "classifier head"
    that turns the final hidden state into a single-channel probability
    map (one probability per grid cell) -- same overall shape as
    Figure 4 in the base paper.

    Input shape:  (batch, T, H, W, input_channels)
    Output shape: (batch, H, W)  -- RAW LOGITS (apply torch.sigmoid to get
                                     a 0-1 probability of crime per cell)
    """

    def __init__(self, input_channels, hidden_channels, num_layers,
                 kernel_size, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(layers)

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x comes in as (batch, T, H, W, C); PyTorch conv wants channels
        # right after batch, so we permute to (batch, T, C, H, W).
        x = x.permute(0, 1, 4, 2, 3)
        batch_size, T, _, H, W = x.shape
        device = x.device

        layer_input = x
        for layer_idx, cell in enumerate(self.cells):
            h, c = cell.init_hidden(batch_size, H, W, device)
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_output = torch.stack(outputs, dim=1)  # (batch, T, hidden, H, W)

            # apply ReLU + batch norm per the paper's architecture (Fig 4)
            layer_output = torch.relu(layer_output)
            # batch norm expects (N, C, H, W); fold T into the batch dim
            bt, tt = layer_output.shape[0], layer_output.shape[1]
            reshaped = layer_output.reshape(bt * tt, *layer_output.shape[2:])
            reshaped = self.batch_norms[layer_idx](reshaped)
            layer_output = reshaped.reshape(bt, tt, *reshaped.shape[1:])

            layer_input = layer_output

        final_hidden = layer_input[:, -1]        # last time step's hidden state
        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)    # (batch, 1, H, W)
        return logits.squeeze(1)                  # (batch, H, W) -- raw logits,
                                                    # NOT passed through sigmoid.
                                                    # We train with BCEWithLogitsLoss
                                                    # (supports class weighting) and
                                                    # apply sigmoid only at prediction
                                                    # time -- see data_utils/predict().


class SimpleLSTM(nn.Module):
    """
    The "plain LSTM" baseline: processes each grid cell's own time series
    independently (no convolution = no awareness of neighboring cells).

    Input shape:  (batch, T, input_channels)   -- one cell's history
    Output shape: (batch,)                      -- RAW LOGITS (apply
                                                    torch.sigmoid to get a
                                                    0-1 probability)
    """

    def __init__(self, input_channels, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]              # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden).squeeze(-1)  # (batch,) raw logits
        return logits   # same reasoning as ConvLSTM -- see comment there
