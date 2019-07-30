import torch.nn as nn
import torch.nn.functional as F

class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()
        # Convolutional layer
        # self.ref = nn.ReflectionPad2d((40,40, 40 , 40))
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 48,
                               kernel_size= 9,stride = 1, padding=4)
        self.in1 = nn.InstanceNorm2d(48)

        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96,
                               kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(96)


        self.conv3 = nn.Conv2d(in_channels= 96, out_channels= 192,
                               kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(192)
        #   5 Residual blocks would be here

        block_channel = 192
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding = 1),
            nn.InstanceNorm2d(block_channel)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=block_channel, out_channels=block_channel,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(block_channel)
        )

        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv4 = nn.Conv2d(in_channels=192, out_channels=96,
                               kernel_size=3, stride=1, padding=1)
        self.in4 = nn.InstanceNorm2d(96)

        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=48,
                               kernel_size=3, stride=1, padding=1)
        self.in5 = nn.InstanceNorm2d(48)
        # self.bn5 = nn.InstanceNorm2d(32)
        self.conv6 = nn.Conv2d(in_channels=48, out_channels=3,
                               kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        # print('original: ', x.shape)

        # x = self.ref(x)
        # print('after padding: ', x.shape)
        x = self.conv1(x)
        x = F.relu(self.in1(x))
        # print('after first layer: ', x.shape)
        x = self.conv2(x)
        x = F.relu(self.in2(x))
        # print('after second layer: ', x.shape)
        x = self.conv3(x)
        x = F.relu(self.in3(x))
        # print('after third layer: ', x.shape)
        residual1 = x
        x = self.block1(x)
        # print('before first block: ', x.shape)
        x = F.relu(x)
        x +=  residual1
        # print('after first block', x.shape)
        # x = x[:,:,2:-2, 2:-2]
        residual2 = x

        # print('before second block: ', x.shape)
        x = self.block2(x)
        x = F.relu(x)
        x += residual2
        # x = x[:, :, 2:-2, 2:-2]
        residual3 = x
        # print('before third block: ', x.shape)

        x = self.block3(x)
        x = F.relu(x)
        x += residual3
        # x = x[:, :, 2:-2, 2:-2]
        # residual4 = x

        # print('after third block: ', x.shape)
        x = self.block4(x)
        x = F.relu(x)

        # print('after fourth block: ', x.shape)

        # print('after all residual block: ', x.shape)
        feature = x

        x = self.upsample1(x)
        # print('after upsample 1: ', x.shape)
        # x = F.relu(self.in4(x))
        #
        # x = x[:, :, 1:, 1:]
        # print('after first deconv layer: ', x.shape)

        x = self.conv4(x)
        x = F.relu(self.in4(x))
        # print('before second upsample 1 &after conv: ', x.shape)
        # x = x[:, :, :-1, :-1]
        # print('after second deconv layer: ', x.shape)

        x = self.upsample2(x)
        # print('after upsample: ', x.shape)
        x = self.conv5(x)
        x = F.relu(self.in5(x))
        # print('before final layer: ', x.shape)
        # print('after third deconv layer: ', x.shape)
        x = self.conv6(x)
        final = F.tanh(x)
        # print('shape of final layer: ', x.shape)
        # x = x[:, :, 4:-4, 4:-4]

        return feature, final
