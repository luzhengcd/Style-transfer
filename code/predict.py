import torch
import numpy as np
import transferNet
import torchvision.transforms as transforms
from PIL import Image

import skvideo.io as io
import cv2
from torchvision.transforms.functional import normalize


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




def predict_pic(model_path, img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_parameters = torch.load(model_path, map_location= device)
    img = imread(img_path).type('torch.FloatTensor')

    model = transferNet.TransferNet()
    model.load_state_dict(model_parameters)
    model.to(device)
    out = model(img)

    new_out = recover_image(out.data.cup().numpy())[0]

    return new_out
    # To show the picture, use:
    # Image.fromarray(new_out)


def predict_video():
    # git checkout test change
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_parameters = torch.load('../model/model_vangogh4.pth', )
    path = '../data/video/2019_NCL_Brand_Essence_Good_to_be_Free.mp4'
    model = transferNet.TransferNet()
    model.load_state_dict(model_parameters)

    model.to(device)
    cap = cv2.VideoCapture(path)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    frame_lst = []
    count = 0

    while (True):
        ret, frame = cap.read()
        count = count + 1
        print(count)

        if ret:
            current_frame = torch.Tensor(frame.transpose(2, 0, 1)) / 255.
            normalized = normalize(current_frame, mean, std)
            new_frame = normalized.reshape((1, *normalized.shape)).to(device)

            out = model(new_frame)
            new_out = recover_image(out.data.cpu().numpy())[0]
            new_out = new_out.reshape((1, *new_out.shape))
            frame_lst.append(new_out)

        else:
            break


    out_video = np.concatenate(frame_lst)
    print(out_video.shape)
    io.vwrite("../outputImage/outputvideo_new.mp4", out_video, outputdict={'-pix_fmt':'yuv420p'})


if __name__ == '__main__':

    predict_video()
