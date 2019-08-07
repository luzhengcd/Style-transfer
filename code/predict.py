import torch
import numpy as np
import transferNet
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timing
from torchvision.transforms.functional import normalize
import argparse
import skvideo.io as io

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


def predict_video(num_frame, device, video_idx):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    model_parameters = torch.load('../model/new_video_vangogh' + str(video_idx)+'.pth')
    path = '../../2019_NCL_Brand_Essence_Good_to_be_Free.mp4'
    model = transferNet.TransferNet()
    model.load_state_dict(model_parameters)

    model.to(device)
    cap = cv2.VideoCapture(path)
    # fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    # out_writer = cv2.VideoWriter("../outputImage/outputvideo_new.avi",
    #                       fourcc, 24.0, (640, 480))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    frame_lst = []
    count = 0
    while (True):

        torch.cuda.empty_cache()

        ret, frame = cap.read()
        count = count + 1

        print(count)
        if ret and count <= num_frame:

            current_frame = torch.Tensor(frame.transpose(2, 0, 1)) / 255.
            normalized = normalize(current_frame, mean, std)
            torch.cuda.empty_cache()
            new_frame = normalized.reshape((1, *normalized.shape)).to(device)

            out = model(new_frame)
            new_out = recover_image(out[1].data.cpu().numpy())
            # out_writer.write(new_out)
            # new_out = new_out.reshape((1, *new_out.shape))
            frame_lst.append(new_out)
        else:
            break
    cap.release()
    # out_writer.release()
    cv2.destroyAllWindows()
    out_video = np.concatenate(frame_lst)
    io.vwrite("../outputImage/new_video_vangogh" + str(video_idx) + ".mp4", out_video, outputdict={'-pix_fmt':'yuv420p'})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-fourcc', type = str, default='mp4v')
    parser.add_argument('-numframe', type = int, default = 30)
    parser.add_argument('-device', type = str, default = 'cpu')
    parser.add_argument('-video_idx', type = int, default = '0')

    args = parser.parse_args()


    predict_video(args.numframe, args.device, args.video_idx)
