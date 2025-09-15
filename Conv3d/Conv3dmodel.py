#3D CNN Model
import torch.nn as nn
class VoxelCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128*16*16*16, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
