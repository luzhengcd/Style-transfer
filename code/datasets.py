import torch
from torch.utils.data import TensorDataset, Dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob

def readData(path_lst):
#     The path is where the pictures are, not a specific picture
    H = 360
    W = 640

    tensor_lst = []

    transform_pipline = transforms.Compose([transforms.Resize([H, W]),
                                            transforms.ToTensor()])

    for path in path_lst:
        with open(path, 'r+b') as f:
            with Image.open(f) as img:
                # The input tensor has to be (C, H, W) in format
                img_new = transform_pipline(img)
                # need to check whether it's a color image or a black white image
                channel = img_new.shape[0]
                if channel == 3:
                    img_new = transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_new)
                    img_new = img_new.unsqueeze(0)
                    img_reflection = F.pad(img_new, pad = (40, 40, 40, 40), mode='reflect')
                    img_reflection = img_reflection.reshape(img_reflection.shape[1:])
                    tensor_lst.append(img_reflection)
                else:
                    continue

    stacked_tensor = torch.stack(tensor_lst)
    stacked_tensor = stacked_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TensorDataset(stacked_tensor)

    return dataset


def crop_array(h, w, img_arr):

    shape = img_arr.shape
    height = shape[0]
    width = shape[1]
    each_h = int((height - h) / 2)
    each_w = int((width - w) / 2)
    return img_arr[each_h : height - each_h, each_w : width - each_w]

def readVideo(path):
    # path is the directory of the video
    img_width = 256
    img_height = 256
    tensor_lst = []
    cap = cv2.VideoCapture(path)
    # count = 0
    while (True):
        ret, frame = cap.read()

        if ret:

            # count = count+1
            # print(count)
            img_new = torch.Tensor(crop_array(img_height, img_width, frame) )
                      # / 255.0
            img_new = img_new.reshape(3, img_height, img_width)

            # img_new = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_new)
            # img_new = img_new.unsqueeze(0)
            # img_reflection = F.pad(img_new, pad=(40, 40, 40, 40), mode='reflect')
            # img_reflection = img_reflection.reshape(img_reflection.shape[1:])
            tensor_lst.append(img_new)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    if len(tensor_lst) % 2 == 0:
        stacked_tensor = torch.stack(tensor_lst)

    else:
        stacked_tensor = torch.stack(tensor_lst[:-1])

    dataset = TensorDataset(stacked_tensor)

    return dataset


def styleImg(path):

#     The path is where the pictures are, not a specific picture
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



def read_flow_occlusion(path):

#     The path is where the pictures are, not a specific picture
    img_size = 256


    transform_pipline = transforms.Compose([transforms.Resize([img_size, img_size]),
                                            ])
    if path.endswith('.png'):

        with open(path, 'r+b') as f:
            with Image.open(f) as img:
                res = transform_pipline(img)
                # img_new = img_new.unsqueeze(0)
    elif path.endswith('.flo'):

        print()

    else:
        print('wrong flow or occlusion file')
    return res


def pair(pic_path, flow_path, occlusion_path):
    pics = glob.glob(pic_path)
    flows = glob.glob(flow_path)
    occlusions = glob.glob(occlusion_path)
    pairs = list(zip(pics, flows, occlusions))

    return pairs


