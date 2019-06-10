import torch
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import transferNet

def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    img_tensor = torch.tensor(img)
    img_tensor = img_tensor.reshape(img_tensor.shape[::-1])
    img_tensor = img_tensor.reshape((1, *img_tensor.shape))
    return img_tensor #returns RGB format


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