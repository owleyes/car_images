import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from vanilla_pytorch_detector import VanillaPytorchDetector, compute_frequency_input


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


class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        self.enc4 = UNetBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        
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
        
        return x


class UNetImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Unet
        self.unet = UNet(3)
        
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
        # Unet
        x = self.unet(x)
        
        # Classifier
        x = self.classifier(x)
        return x


class TwoStreamUNetClassifier(nn.Module):
    def __init__(self):
        super(TwoStreamUNetClassifier, self).__init__()
        self.spatial_stream = UNet(3)
        self.frequency_stream = UNet(1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, rgb):
        freq = compute_frequency_input(rgb)
        spatial_features = self.spatial_stream(rgb)
        freq_features = self.frequency_stream(freq)
        fused = torch.cat((spatial_features, freq_features), dim=1)
        out = self.classifier(fused)
        return out


class UNetSpatialDetector(VanillaPytorchDetector):
    def __init__(self):
        super().__init__()

    def get_model(self) -> nn.Module:
        self.model = UNetImageClassifier()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        return self.model


class UNetFreqDetector(VanillaPytorchDetector):
    def __init__(self):
        super().__init__()

    def get_model(self) -> nn.Module:
        self.model = TwoStreamUNetClassifier()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        return self.model
        