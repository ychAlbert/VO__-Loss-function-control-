import torch
import torch.nn as nn

class CorrectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),  # 输入: RGB(3) + 光流(2)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 输出Δu, Δv
        )
    
    def forward(self, img_patch, flow):
        x = torch.cat([img_patch, flow], dim=1)
        x = self.conv_layers(x)
        return self.fc(x)
