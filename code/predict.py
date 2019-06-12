import torch
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import transferNet
import torchvision.transforms as transforms
from PIL import Image


#
# def imread(path):
#
#
#
#     img = scipy.misc.imread(path).astype(np.float)
#     img_tensor = torch.tensor(img)
#     img_tensor = img_tensor.reshape(img_tensor.shape[::-1])
#     img_tensor = img_tensor.reshape((1, *img_tensor.shape))
#     return img_tensor #returns RGB format

def imread(path):

#     The path is where the pictures are, not a specific picture

    transform_pipline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
    with open(path, 'r+b') as f:
        with Image.open(f) as img:
            img_new = transform_pipline(img)
            img_new = img_new.reshape((1, *img_new.shape))
    return img_new

if __name__ == '__main__':

    model_parameters = torch.load('../model.pth', map_location='cpu')
    img_path = ''
    img = imread(img_path).type('torch.FloatTensor')

    model = transferNet.TransferNet()
    model.load_state_dict(model_parameters)
    out = model(img)
    out = out.reshape(out.shape[1:][::-1])
    final = np.array(out.data).astype(np.uint8)
    plt.imshow(final)