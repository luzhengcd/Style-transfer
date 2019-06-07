import torchvision.models as models
import torch
import torch.nn as nn


# sometimes * before a tensor can flatten the tensor.

# The layer for calculating the style loss:
# relu 1-2, 2-2, 3-3. 4-3
# The layer for the feature loss:
# relu 3-3
original_model = models.vgg16(pretrained=True)

class Relu_12(nn.Module):
    def __init__(self):
        super(Relu_12, self).__init__()
        self.features = original_model.features[:-27]

    def forward(self, x):
        # print('BEFORE RELU 12 SHAPE: ', x.shape)

        x = self.features(x)
        # print('RELU 12 SHAPE: ', x.shape)
        return x


class Relu_22(nn.Module):
    def __init__(self):
        super(Relu_22, self).__init__()
        self.features = original_model.features[:-22]

    def forward(self, x):
        # print('BEFORE RELU 22 SHAPE: ', x.shape )

        x = self.features(x)
        # print('RELU 22 SHAPE: ', x.shape )
        return x

class Relu_33(nn.Module):
    def __init__(self):
        super(Relu_33, self).__init__()
        self.features = original_model.features[:-15]

    def forward(self, x):
        # print('BEFORE RELU 33 SHAPE: ', x.shape)
        x = self.features(x)
        # print('RELU 33 SHAPE: ', x.shape)

        return x

class Relu_43(nn.Module):
    def __init__(self):
        super(Relu_43, self).__init__()
        self.features = original_model.features[:-8]

    def forward(self, x):
        # print('BEFORE RELU 43 SHAPE: ', x.shape)
        x = self.features(x)
        # print('RELU 43 SHAPE: ', x.shape)

        return x

