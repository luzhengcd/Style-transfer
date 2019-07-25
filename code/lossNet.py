import torchvision.models as models
import torch.nn as nn

'''
    This script builds the loss network based on a pre-trained vgg16 in pytorch.
    The paper extracts output from different layers to represent the style and content.
    You can visualize the output using optimization approach described in the paper. 
    Visualization may help you decide what kind of layer combination can give you best result.
    
    I slice vgg16 and code out separate networks class for each layer. There must be a simpler 
    way to do it. You may need to do some research if you want to make it more succinct.
     
    The style layer:
        relu 1-2, 2-2, 3-3, 4-3
    The content layer:
        relu 3-3
'''

original_model = models.vgg16(pretrained=True)

class Relu_12(nn.Module):
    def __init__(self):
        super(Relu_12, self).__init__()
        self.features = original_model.features[:-27]

    def forward(self, x):
        x = self.features(x)
        return x


class Relu_22(nn.Module):
    def __init__(self):
        super(Relu_22, self).__init__()
        self.features = original_model.features[:-22]

    def forward(self, x):
        x = self.features(x)
        return x

class Relu_33(nn.Module):
    def __init__(self):
        super(Relu_33, self).__init__()
        self.features = original_model.features[:-15]

    def forward(self, x):
        x = self.features(x)
        return x

class Relu_43(nn.Module):
    def __init__(self):
        super(Relu_43, self).__init__()
        self.features = original_model.features[:-8]

    def forward(self, x):
        x = self.features(x)
        return x

