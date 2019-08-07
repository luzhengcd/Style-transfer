from lossNet import *
# from utils import Gram, crop_array
import utils
import numpy as np
import cv2
import IO

"""
To calculate the loss, the following steps need to be implemented:
    - Perform a forward pass with transferNet. The output of the transferNet
    will be the input of lossNet
    - Get the output of specified layers of lossNet.
    - Calculate the style loss and the content loss.
    - Perform the optimization
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

relu12 = Relu_12()
relu22 = Relu_22()
relu33 = Relu_33()
relu43 = Relu_43()


relu12.to(device)
relu22.to(device)
relu33.to(device)
relu43.to(device)

def styleLoss(y_s, y_hat, criterion):
    """
    :param y_s: style target, which is style picture
    :param y_hat: output of the transferNet
    :return: return the style loss
    Note that y_hat and returned value should be a variable type
    cuz we need to update the variable to minimize the cost function
    """
    # perform a forward pass for both y_s and y_hat

    """
    The dimension of outputs from each relu layer:
        - Relu 12: 64
        - Relu 22: 128
        - Relu 33: 256
        - Relu 43: 512
    """

    output12_ys = relu12(y_s)
    output12_hat = relu12(y_hat)
    output22_ys = relu22(y_s)
    output22_hat = relu22(y_hat)
    output33_ys = relu33(y_s)
    output33_hat = relu33(y_hat)
    output43_ys = relu43(y_s)
    output43_hat = relu43(y_hat)

    #   calculate the gram matrix for each layer
    #   right now, we assume the outputs have a shape of [3,.., ..]

    loss1 = criterion(utils.Gram(output12_hat), utils.Gram(output12_ys))
    loss2 = criterion(utils.Gram(output22_hat), utils.Gram(output22_ys))
    loss3 = criterion(utils.Gram(output33_hat), utils.Gram(output33_ys))
    loss4 = criterion(utils.Gram(output43_hat), utils.Gram(output43_ys))

    total_style_loss = loss1 + loss2 + loss3 + loss4
    total_style_loss = total_style_loss
    return total_style_loss

def contentLoss(y_c, y_hat, criterion):

    # y_c_new = y_c[:,:,40:-40, 40:-40]

    output22_yc = relu22(y_c)
    output22_hat = relu22(y_hat)

    # squared, normalized Euclidean distance between feature representations
    # CHW = torch.prod(torch.tensor(y_c_new.shape[1:]))
    # The shape of the output22_yc is [#pic, ...]

    hat_flatten = output22_hat.reshape(output22_hat.shape[0], -1)
    yc_flatten = output22_yc.reshape(output22_yc.shape[0], -1)

    loss_content = criterion(hat_flatten, yc_flatten)
    return loss_content


def reshape_batch(batch, batch_size):
    res = batch.reshape((batch_size, *batch.shape[2:], batch.shape[1]))
    return res



# def warp_flow(batch, flow_lst):
#     # note that img here is also a batch, which contains 2 images with the format of (#frame, C, H, W)
#     # after reshape, the shape of the batch becomes (#frame, H, W, C), and the shape of flow tensor is (#frame, H, W, grad)
#
#     batch_size = batch.shape[0]
#     # batch size would be 2
#     #
#
#     batch_reshape =reshape_batch(batch, batch_size)
#     h, w = flow_lst[0].shape[:2]
#
#
#     for i in range(batch_size):
#         flow_lst[i][:, :, 0] += np.arange(h)[:, np.newaxis]
#         flow_lst[i][:, :, 1] += np.arange(w)
#
#
#     # cv2.remap(batch_reshape[0].data.numpy(), flow_lst[0], None, cv2.INTER_LINEAR)
#
#     warped_lst = [torch.Tensor(cv2.remap(batch_reshape[i].cpu().data.numpy(), flow_lst[i],
#                         None, cv2.INTER_LINEAR)) for i in range(batch_size)]
#
#     warped = torch.stack(warped_lst)
#     res = warped.reshape((batch_size, batch.shape[1], h, w))
#
#     return res


def warp_flow(batch, flow_path):
    # note that img here is also a batch, which contains 2 images with the format of (#frame, C, H, W)
    # after reshape, the shape of the batch becomes (#frame, H, W, C), and the shape of flow tensor is (#frame, H, W, grad)

    # batch size would be 2
    flow = IO.read(flow_path)[:, :, :2]
    num_frame, C, H, W = batch.shape
    flow = utils.crop_array(H, W, flow)

    pic_after_arr = batch[1].data.cpu().numpy().reshape(H, W, 3)

    flow[:, :, 0] += np.arange(W)
    flow[:, :, 1] += np.arange(H)[:, np.newaxis]

    res = cv2.remap(pic_after_arr, flow, None, cv2.INTER_LINEAR)

    return res



    # back to the size of a normal batch
    # res_reshape = torch.Tensor(res.reshape((1, 3, H, W)))

    # return res_reshape
    # return res


def temporalF(batch, flow_path, occlusion_path, criterion):

    pre = batch[0]

    # f = open(flow_path, 'rb')

    # header = f.read(4)
    # if header.decode("utf-8") != 'PIEH':
    #     print('Flow file header does not contain PIEH')
    #     return 0
    # else:

    warped = warp_flow(batch, flow_path)

    num_frame, C, H, W = batch.shape
    # need to add mask later
    # occlusion = IO.read(occlusion_path)

    # the shape of batch (#frame, C, H, W)

    pre_reshape = pre.reshape((H, W, 3)).cpu().numpy()

    mask = utils.crop_array(H, W, -(IO.read(occlusion_path) / 255 - 1)).reshape((H, W, 1))

    masked_warp = torch.Tensor((mask * warped))
    masked_pre = torch.Tensor((mask * pre_reshape))

    pre_final = masked_pre.to(device)
    warp_final = masked_warp.to(device)

    return criterion(pre_final, warp_final)


def temporalO(O_current, O_pre, criterion, flow):

    # For BGR image, it returns an array of Blue, Green, Red values. (B, G, R)
    O_warp = warp_flow(O_pre, flow)
    # I_warp = warp_flow(I_pre, flow)
    # print(O_current.shape)
    # print(O_warp.shape)
    O_warp = O_warp.to(device)
    # I_warp = I_warp.to(device)
    # temp1 = O_current -  O_warp
    # temp2 = I_current - I_warp

    # 3 channels:
    # b = temp2[:, 0, :, :]
    # g = temp2[:, 1, :, :]
    # r = temp2[:, 2, :, :]
    # based on the paper, calculate the relative luminance

    # Y = 0.2126 * r + 0.7152*g + 0.0722*b
    # (#frame, C, H, W)
    # Y = Y.reshape((2, 1, temp1.shape[2], temp1.shape[3]))
    # print('temp shape: ', temp2.shape)
    # print('first shape', temp1.shape)
    # print('second shape', Y.shape)

    # return criterion(temp1, Y)
    return criterion(O_warp, O_current)
