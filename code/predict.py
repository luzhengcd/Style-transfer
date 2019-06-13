import torch
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import transferNet
import torchvision.transforms as transforms
from PIL import Image



def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)



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

    new_out = recover_image(out.data.cup().numpy())[0]
    Image.fromarray(new_out)

