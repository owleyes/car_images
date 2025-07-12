import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vanilla_pytorch_detector import VanillaPytorchDetector


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        

class UNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = F.max_pool2d(x1, 2)
        
        x2 = self.enc2(x)
        x = F.max_pool2d(x2, 2)
        
        x3 = self.enc3(x)
        x = F.max_pool2d(x3, 2)
        
        x4 = self.enc4(x)
        x = F.max_pool2d(x4, 2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Classifier
        x = self.classifier(x)
        return x


class UNetDetector(VanillaPytorchDetector):
    def __init__(self):
        super().__init__()

    def get_model(self) -> nn.Module:
        self.model = UNetClassifier()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        return self.model
        