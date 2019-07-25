import torch
import numpy as np
import transferNet
import torchvision.transforms as transforms
from PIL import Image
import argparse


'''
    This script is to yield stylized image given a trained model and an arbitrary content picture.
    
    It contains 2 functions:
        recover_image: recall that we normalized the input in the training phase. This function, 
                       as its name indicates, is to recover image by multiplying std and adding mean 
                       so that it can be properly displayed.
        imread:        Same as the one in datasets.py. It is used to read in a content picture you 
                       want to transfer the style onto.      
'''


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

    transform_pipline = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
    with open(path, 'r+b') as f:
        with Image.open(f) as img:
            img_new = transform_pipline(img)
            img_new = img_new.reshape((1, *img_new.shape))
    return img_new


def predict(modelPath, imgPath, savePath):
    # load the model parameters
    model_parameters = torch.load('../'+ modelPath, map_location='cpu')
    img_path = imgPath
    img = imread(img_path).type('torch.FloatTensor')

    model = transferNet.TransferNet()
    model.load_state_dict(model_parameters)
    out = model(img)
    # try the following two ways to recover image and use the one working
    new_out = recover_image(out.data.cup().numpy())[0]
    # new_out = recover_image(out[0].data.cup().numpy())

    # save the image
    new_im = Image.fromarray(new_out[0, :], 'RGB')
    new_im.save('../'+savePath)


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelPath', type=str, default='model.pth')
    parser.add_argument('-imgPath', type=str, default='imgPath')
    parser.add_argument('-savePath', type = str, default = 'default.png')
    args = parser.parse_args()

    predict(args.ModelPath, args.imgPath, args.savePath)