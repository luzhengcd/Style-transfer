import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()
        # Convolutional layer

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32,
                               kernel_size= 9,stride = 1, padding=4)
        self.bn1 = nn.InstanceNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels=128,
                               kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.InstanceNorm2d(128)
        #   5 Residual blocks would be here

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding = 1),
            nn.InstanceNorm2d(128)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(128)
        )

        self.conv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=2)
        self.bn4 = nn.InstanceNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=2)
        self.bn5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(in_channels=32, out_channels=3,
                               kernel_size=9, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        residual1 = x
        x = self.block1(x)

        x +=  residual1
        x = x[:,:,2:-2, 2:-2]
        residual2 = x
        x = self.block2(x)
        x += residual2
        x = x[:, :, 2:-2, 2:-2]
        residual3 = x
        x = self.block3(x)
        x += residual3
        x = x[:, :, 2:-2, 2:-2]

        residual4 = x
        x = self.block4(x)
        x += residual4
        x = x[:, :, 2:-2, 2:-2]

        residual5 = x

        x = self.block5(x)
        x += residual5
        x = x[:,:, 2:-2, 2:-2]

        x = self.conv4(x)

        x = F.relu(self.bn4(x))
        x = x[:, :, 1:, 1:]

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = x[:, :, 1:, 1:]
        x = self.conv6(x)

        x = F.relu(x)
        x = x[:, :, 4:-4, 4:-4]

        return x
