import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
import glob
import io
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def readData(path):
#     The path is where the pictures are, not a specific picture
    pic_path = glob.glob(path)
    img_size = 256
    tensor_lst = []
    transform_pipline = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.ToTensor()])
    for path in pic_path:
        with open(path, 'r+b') as f:
            with Image.open(f) as img:
                img_new = transform_pipline(img)
                # need to check whether it's a color image or a black white image
                channel = img_new.shape[0]
                if channel == 3:
                    img_new = img_new.unsqueeze(0)
                    img_reflection = F.pad(img_new, pad = (40, 40, 40, 40), mode='reflect')
                    img_reflection = img_reflection.reshape(img_reflection.shape[1:])
                    tensor_lst.append(img_reflection)
                else:
                    continue

    stacked_tensor = torch.stack(tensor_lst)
    stacked_tensor = stacked_tensor.cuda()
    dataset = TensorDataset(stacked_tensor)

    return dataset


def styleImg(path):

#     The path is where the pictures are, not a specific picture
    img_size = 256

    transform_pipline = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.ToTensor()])
    with open(path, 'r+b') as f:
        with Image.open(f) as img:
            img_new = transform_pipline(img)
            img_new = img_new.unsqueeze(0)
    return img_new


