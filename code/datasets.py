import torch
from torch.utils.data import TensorDataset, Dataset
from PIL import Image
import torchvision.transforms as transforms


'''
    This script aims to load picture data in the format supported by pytorch when training a neural network. 
    It contains 2 functions:
        readData: Load the content images dataset which usually contains many pictures with different contents
        readImage: Load 1 style image     
'''

def readData(path_lst):

    '''
    :param path_lst: a list of picture path
    :return: a tensor datasets which contains all pictures in the form of tensor
    '''

    # All the image are resized to 256*256. You can also change img_width and img_height depending on your needs.
    img_width = 256
    img_height = 256
    tensor_lst = []

    # transform_pipline takes a series of operation you want to do to a image , and do it at once when you call the pipeline.
    # check the documentation for other operations it supports.

    transform_pipline = transforms.Compose([transforms.Resize([img_height, img_width]),
                                            transforms.ToTensor()])

    for path in path_lst:
        with open(path, 'r+b') as f:
            with Image.open(f) as img:

                img_new = transform_pipline(img)

                channel = img_new.shape[0]
                # There may be black-white pictures in some dataset. The 'if' here is to rule out those black-white picture and only keep color ones.
                if channel == 3:
                    # By default, all input pictures you feed into a pytorch pre-trained model need to be normalized. You can find the mean and std in
                    # pytorch official documentation.
                    img_new = transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_new)
                    tensor_lst.append(img_new)
                else:
                    continue

    stacked_tensor = torch.stack(tensor_lst)

    stacked_tensor = stacked_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TensorDataset(stacked_tensor)

    return dataset


def styleImg(path):

    '''
    :param path: the path to the image picture
    :return: a tensor type image picture
    '''

    img_size = 256
    transform_pipline = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
    with open(path, 'r+b') as f:
        with Image.open(f) as img:
            img_new = transform_pipline(img)
            img_new = img_new.unsqueeze(0)
    return img_new


