"""MCDANNNet: Multi-Channel Dense Attention Neural Network for ECG classification.

Copied verbatim from ecg/ECGModel/MCDANN.py (the pure nn.Module parts only).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        # Cải tiến: Thêm dropout và better initialization
        reduced_dim = max(in_channels // reduction, 4)
        self.fc1 = nn.Linear(in_channels, reduced_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(reduced_dim, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=8, kernel_sizes=[5, 3]):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm1d(in_channels)
        # Cải tiến: Thêm bias=False và dropout
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=kernel_sizes[0],
                               padding="same", bias=False)
        self.dropout1 = nn.Dropout1d(0.1)

        self.bn2 = nn.BatchNorm1d(in_channels + growth_rate)
        self.conv2 = nn.Conv1d(in_channels + growth_rate, growth_rate, kernel_size=kernel_sizes[1],
                               padding="same", bias=False)
        self.dropout2 = nn.Dropout1d(0.1)

    def forward(self, x):
        # First Composite Function
        out = self.bn1(x)
        out = self.lrelu(out)
        out = self.conv1(out)
        if self.training:
            out = self.dropout1(out)
        x = torch.cat([x, out], dim=1)

        # Second Composite Function
        out = self.bn2(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.training:
            out = self.dropout2(out)
        x = torch.cat([x, out], dim=1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        # Cải tiến: Thêm batch norm và activation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.lrelu = nn.LeakyReLU(0.01)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.pool(x)
        return x


class DACB(nn.Module):
    def __init__(self):
        super(DACB, self).__init__()
        # Cải tiến: Multi-scale initial convolution
        self.conv1_3 = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(1, 4, kernel_size=5, padding=2)
        self.conv1_7 = nn.Conv1d(1, 4, kernel_size=7, padding=3)
        self.conv1_9 = nn.Conv1d(1, 4, kernel_size=9, padding=4)
        self.initial_bn = nn.BatchNorm1d(16)
        self.initial_lrelu = nn.LeakyReLU(0.01)

        self.dense1 = DenseBlock(16)
        self.transition1 = TransitionLayer(32)
        self.se1 = SEBlock(64)

        self.dense2 = DenseBlock(64)
        self.transition2 = TransitionLayer(80)
        self.se2 = SEBlock(64)

        # Cải tiến: Additional dense block for deeper features
        self.dense3 = DenseBlock(64)
        self.final_conv = nn.Conv1d(80, 64, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm1d(64)

        # Cải tiến: Better skip connection
        self.skip = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01)
        )
        self.lrelu = nn.LeakyReLU(0.01)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Cải tiến: Multi-scale feature extraction
        f1 = self.conv1_3(x)
        f2 = self.conv1_5(x)
        f3 = self.conv1_7(x)
        f4 = self.conv1_9(x)
        x = torch.cat([f1, f2, f3, f4], dim=1)
        x = self.initial_bn(x)
        x = self.initial_lrelu(x)

        # Store for skip connection
        skip_input = x

        d = self.dense1(x)
        d = self.transition1(d)
        d = self.se1(d)

        d = self.dense2(d)
        d = self.transition2(d)
        d = self.se2(d)

        # Cải tiến: Additional processing
        d = self.dense3(d)
        d = self.final_conv(d)
        d = self.final_bn(d)
        d = self.lrelu(d)

        skip = self.skip(skip_input)

        # Cải tiến: Residual connection instead of concatenation
        if d.size(-1) != skip.size(-1):
            skip = F.interpolate(skip, size=d.size(-1), mode='linear', align_corners=False)

        out = d + skip  # Residual connection
        out = self.lrelu(out)
        out = self.gap(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=12):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1)]


class MCDANNNet(nn.Module):
    def __init__(self, num_classes):
        super(MCDANNNet, self).__init__()
        self.channels = nn.ModuleList([DACB() for _ in range(12)])  # 12 module per 12 leads

        # Cải tiến: Positional encoding for leads
        self.positional_encoding = PositionalEncoding(d_model=64, max_seq_length=12)

        # Cải tiến: Cross-lead attention mechanism
        self.lead_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=1, dropout=0.1, batch_first=True
        )

        # Cải tiến: Enhanced classifier with feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Cải tiến: Adaptive downsampling
        x = x[:, :, ::2]  # Downsample 0.5

        # Cải tiến: Enhanced normalization (per-lead)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + 1e-8)

        features = []
        for i, channel in enumerate(self.channels):
            lead = x[:, i, :].unsqueeze(1)
            feat = channel(lead).squeeze(-1)
            features.append(feat)

        # Cải tiến: Stack features for attention [batch, num_leads, 64]
        stacked_features = torch.stack(features, dim=1)

        # Cải tiến: Add positional encoding to help attention understand lead positions
        stacked_features = self.positional_encoding(stacked_features)

        # Cải tiến: Apply cross-lead attention
        attended_features, _ = self.lead_attention(
            stacked_features, stacked_features, stacked_features
        )

        # Cải tiến: Combine original and attended features
        enhanced_features = stacked_features + attended_features

        # Flatten for classification
        combined = enhanced_features.view(batch_size, -1)
        return self.classifier(combined)
