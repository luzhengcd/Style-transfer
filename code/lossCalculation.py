import torch
import torch.nn as nn
from lossNet import *
import utils
import numpy as np
from torch.autograd import Variable

"""
To calculate the loss, the following steps need to be implemented:
    - Perform a forward pass with transferNet. The output of the transferNet
    will be the input of lossNet
    - Get the output of specified layers of lossNet. 
    - Calculate the style loss and the content loss.
    - Perform the optimization
"""



def styleLoss(y_s, y_hat, criterion):
    """
    :param y_s: style target, which is style picture
    :param y_hat: output of the transferNet
    :return: return the style loss
    Note that y_hat and returned value should be a variable type
    cuz we need to update the variable to minimize the cost function
    """
    #   perform a forward pass for both y_s and y_hat
    relu12 = Relu_12()
    relu22 = Relu_22()
    relu33 = Relu_33()
    relu43 = Relu_43()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    relu12.to(device)
    relu22.to(device)
    relu33.to(device)
    relu43.to(device)

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
    # print(utils.Gram(output12_ys))
    # print(utils.Gram(output12_hat))
    loss1 = criterion(utils.Gram(output12_hat), utils.Gram(output12_ys))
    loss2 = criterion(utils.Gram(output22_hat), utils.Gram(output22_ys))
    loss3 = criterion(utils.Gram(output33_hat), utils.Gram(output33_ys))
    loss4 = criterion(utils.Gram(output43_hat), utils.Gram(output43_ys))

    total_style_loss = loss1 + loss2 + loss3 + loss4
    total_style_loss = total_style_loss.to('cuda' if torch.cuda.is_available() else 'cpu')
    return total_style_loss

# def help_style(y_s, y_hat, ):
#
#     gram_y = utils.Gram(y_s)
#     gram_y_hat = utils.Gram(y_hat)
#
#     temp_gram = (gram_y - gram_y_hat) ** 2
#     loss = torch.sum(temp_gram)
#     return loss

def contentLoss(y_c, y_hat, criterion):
    # y_c_new = y_c[:,:,40:-40, 40:-40]
    relu22 = Relu_22()

    relu22.to('cuda' if torch.cuda.is_available() else 'cpu')


    output22_yc = relu22(y_c)
    output22_hat = relu22(y_hat)
    # squared, normalized Euclidean distance between feature representations

    # CHW = torch.prod(torch.tensor(y_c_new.shape[1:]))
    # The shape of the output22_yc is [#pic, ...]

    hat_flatten = output22_hat.reshape(output22_hat.shape[0], -1)
    yc_flatten = output22_yc.reshape(output22_yc.shape[0], -1)

    # mean_hat = torch.mean(hat_flatten)
    # mean_yc = torch.mean(yc_flatten)


    # loss_content = 0.5 * torch.sum(((hat_flatten - mean_hat) - (yc_flatten - mean_hat)) ** 2) / \
    #                ((torch.sum((hat_flatten -mean_hat)**2 ) + torch.sum((yc_flatten - mean_yc) ** 2)) * CHW)
    # print(hat_flatten.shape)
    # print(yc_flatten.shape)

    loss_content = criterion(hat_flatten, yc_flatten)
    return loss_content
