import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotoacousticQualityNet(nn.Module):
    def __init__(self, in_channels=1, num_fc_units=128, dropout_rate=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 128×128 → 8×8 after 4x pooling
        self.flatten_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(self.flatten_dim, num_fc_units)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(num_fc_units, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv4(x)))  # 16 -> 8

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)