import torch
import torch.nn as nn
from lossNet import *
# from utils import *
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

# The layer for calculating the style loss:
# relu 1-2, 2-2, 3-3. 4-3
# The layer for the feature loss:
# relu 3-3

#
# def styleLoss(y_s, y_hat, org_model):
#
#     """
#     :param y_s: style target, which is style picture
#     :param y_hat: output of the transferNet
#     :return: return the style loss
#     Note that y_hat and returned value should be a variable type
#     cuz we need to update the variable to minimize the cost function
#     """
# #   perform a forward pass for both y_s and y_hat
#
#     relu12 = Relu_12()
#     relu22 = Relu_22()
#     relu33 = Relu_33()
#     relu43 = Relu_43()
#
#     output12_ys = relu12(y_s)
#     output12_hat = relu12(y_hat)
#     output22_ys = relu22(y_s)
#     output22_hat = relu12(y_hat)
#     output33_ys = relu33(y_s)
#     output33_hat = relu33(y_hat)
#     output43_ys = relu43(y_s)
#     output43_hat = relu43(y_hat)
#
# #   calculate the gram matrix for each layer
# #   right now, we assume the outputs have a shape of [3,.., ..]
# #   may change depending on the actual size of y and y_hat as the codes develop
#
#     loss1 = help_style(output12_hat, output12_ys)
#     loss2 = help_style(output22_hat, output22_ys)
#     loss3 = help_style(output33_hat, output33_ys)
#     loss4 = help_style(output43_hat, output43_ys)
#
#     total_style_loss = loss1 + loss2 + loss3 + loss4
#
#     return total_style_loss
#
#
# def help_style(y_s, y_hat):
#
#     gram_y = Gram(y_s)
#     gram_y_hat = Gram(y_hat)
#     loss = torch.sum(np.square(gram_y - gram_y_hat))
#     return loss
#
#
# def contentLoss(y_c, y_hat, org_model):
#
#     relu33 = Relu_33()
#     output33_yc = relu33(y_c)
#     output33_hat = relu33(y_hat)
#     # squared, normalized Euclidean distance between feature representations
#     loss_content = np.sum(np.square(output33_hat - output33_yc))/np.prod(y_c.shape)
#
#     return loss_content
#
#
# def totalLoss(y_c, y_s, y_hat, org_model, weight = [0.5, 0.5]):
#     print(type(y_hat))
#     y_variable = Variable(y_hat, requires_grad = True)
#     y_c_org = y_c[:,:, 40:-40, 40:-40]
#     content = contentLoss(y_c_org, y_variable, org_model)
#     style = styleLoss(y_s, y_variable, org_model)
#     total = weight[0] * content + weight[1] * style
#
#     return total
#
#
#
#
#
class TotalLoss(torch.nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
    # so basically, the x in foward function should be the input so that the backward() function can recognize it as the
    # parameters need to be updated???

    def forward(self, y_c, y_s, y_hat):
        y_s = y_s.cuda()
        y_hat = y_hat.cuda()
        y_hat_var = torch.autograd.Variable(y_hat)
        y_c_var = torch.autograd.Variable(y_c)
        y_s_var = torch.autograd.Variable(y_s)
        weight = [0.5, 0.5]

        content = self.contentLoss(y_c_var, y_hat_var)
        style = self.styleLoss(y_s_var, y_hat_var)

        content = content.cuda()
        style = style.cuda()
        total = weight[0] * content + weight[1] * style
        return total



    def styleLoss(self, y_s, y_hat):
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

        loss1 = self.help_style(output12_hat, output12_ys)
        loss2 = self.help_style(output22_hat, output22_ys)
        loss3 = self.help_style(output33_hat, output33_ys)
        loss4 = self.help_style(output43_hat, output43_ys)

        total_style_loss = loss1 + loss2 + loss3 + loss4
        total_style_loss = total_style_loss.cuda()
        return total_style_loss

    def help_style(self, y_s, y_hat):

        gram_y = utils.Gram(y_s)
        gram_y_hat = utils.Gram(y_hat)

        # print(gram_y_hat.shape)
        # print(gram_y.shape)

        temp_gram = (gram_y - gram_y_hat) ** 2
        loss = torch.sum(temp_gram)
        return loss

    def contentLoss(self, y_c, y_hat):
        y_c_new = y_c[:,:,40:-40, 40:-40]
        relu33 = Relu_33()

        relu33.to('cuda')

        output33_yc = relu33(y_c_new)
        output33_hat = relu33(y_hat)
        # squared, normalized Euclidean distance between feature representations
        # print(y_c.shape)
        # print(y_hat.shape)
        # print(output33_hat.shape)
        # print(output33_yc.shape)
        # Damn, another dimension issue
        CHW = torch.prod(torch.tensor(y_c_new.shape[1:]))
        loss_content = torch.sum((output33_hat - output33_yc) ** 2) / CHW

        return loss_content

