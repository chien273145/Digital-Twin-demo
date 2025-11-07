import torch
import torch.nn as nn

class DigitalTwinModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal architecture — thay bằng kiến trúc thực tế
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)
