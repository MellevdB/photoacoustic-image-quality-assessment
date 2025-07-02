import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class PhotoacousticQualityNet(nn.Module):
    def __init__(self, in_channels=1, conv_filters=[32, 64, 128, 256], num_fc_units=128, dropout_rate=0.3):
        super().__init__()
        assert len(conv_filters) == 4, "Expected 4 convolutional layer filter sizes"

        self.conv1 = nn.Conv2d(in_channels, conv_filters[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(conv_filters[2], conv_filters[3], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = conv_filters[3] * 8 * 8  # After 4 poolings on 128×128

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
    
class PhotoacousticQualityNetBN(nn.Module):
    def __init__(self, in_channels=1, conv_filters=[32, 64, 128, 256], num_fc_units=128):
        super().__init__()
        assert len(conv_filters) == 4, "Expected 4 convolutional layers"

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_filters[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(conv_filters[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_filters[1]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_filters[2]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv_filters[2], conv_filters[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_filters[3]),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_dim = conv_filters[3] * 8 * 8

        self.fc1 = nn.Linear(self.flatten_dim, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units)
        self.fc3 = nn.Linear(num_fc_units, 1)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class PhotoacousticQualityNetMulti(nn.Module):
    def __init__(self, in_channels=1, conv_filters=[32, 64, 128, 256], num_fc_units=128, dropout_rate=0.3, num_outputs=3):
        super().__init__()
        assert len(conv_filters) == 4, "Expected 4 convolutional layer filter sizes"

        self.conv1 = nn.Conv2d(in_channels, conv_filters[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(conv_filters[2], conv_filters[3], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = conv_filters[3] * 8 * 8  # After 4 poolings on 128×128

        self.fc1 = nn.Linear(self.flatten_dim, num_fc_units)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(num_fc_units, num_outputs)  # Predict num_outputs metric scores

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
    

class IQDCNN(nn.Module):
    def __init__(self, in_channels=1, conv_filters=[32, 32, 32, 32], num_fc_units=1024, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_filters[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv_filters[1], conv_filters[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(conv_filters[2], conv_filters[2], kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(conv_filters[3], conv_filters[3], kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Dynamically compute flattened dimension after conv+pool layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 128, 128)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, num_fc_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(num_fc_units, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = F.relu(self.fc3(x)); x = self.dropout3(x)
        return self.fc4(x)
    
class EfficientNetIQA(nn.Module):
    def __init__(self, pretrained=True, in_channels=1, num_fc_units=128):
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)

        # Patch the first conv layer to accept grayscale input
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Replace classifier for regression (and allow flexible hidden size)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1)
        )

    def forward(self, x):
        return self.model(x)
    


class IQDCNNMulti(nn.Module):
    def __init__(self, in_channels=1, conv_filters=[32, 32, 32, 32], num_fc_units=1024, dropout_rate=0.3, num_outputs=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_filters[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv_filters[1], conv_filters[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(conv_filters[2], conv_filters[2], kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(conv_filters[3], conv_filters[3], kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Dynamically determine flatten_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 128, 128)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            self.flatten_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flatten_dim, num_fc_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(num_fc_units, num_fc_units)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(num_fc_units, num_outputs)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = F.relu(self.fc3(x)); x = self.dropout3(x)
        return self.fc4(x)
    
class EfficientNetIQAMulti(nn.Module):
    def __init__(self, pretrained=True, num_outputs=3, in_channels=1, num_fc_units=128):
        super().__init__()
        self.model = efficientnet_b0(pretrained=pretrained)
        
        # Patch first conv to accept grayscale
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Replace classifier for multi-output regression
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, num_outputs)
        )

    def forward(self, x):
        return self.model(x)