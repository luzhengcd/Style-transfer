from lossNet import *
import utils

'''
    This script builds on lossNet, and is used to calculate the loss when training the network
    To calculate the loss, the following steps need to be implemented:
        - Perform a forward pass through transferNet. The output of the transferNet will be the input of lossNet.
        - Get the output of specified layers of lossNet. Different layer corresponds to different style or content
          representation. Check the paper for more details.
        - Calculate the style loss and the content loss.
        - Perform the optimization, which will be covered in train.py

'''


def styleLoss(y_s, y_hat, criterion):

    """
    :param y_s: style target, which is style picture
    :param y_hat: output of content picture passing through transferNet
    :return: the style loss
    """

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

    # get the gram matrix for each layer

    loss1 = criterion(utils.Gram(output12_hat), utils.Gram(output12_ys))
    loss2 = criterion(utils.Gram(output22_hat), utils.Gram(output22_ys))
    loss3 = criterion(utils.Gram(output33_hat), utils.Gram(output33_ys))
    loss4 = criterion(utils.Gram(output43_hat), utils.Gram(output43_ys))

    # I simply add up the loss from each layer with equal weight to get the total loss.
    # You can also adjust the weights.
    total_style_loss = loss1 + loss2 + loss3 + loss4
    total_style_loss = total_style_loss.to('cuda' if torch.cuda.is_available() else 'cpu')
    return total_style_loss


def contentLoss(y_c, y_hat, criterion):
    '''
    :param y_c: content picture
    :param y_hat: output of content picture passing through transferNet
    :param criterion: MSE defined in pytorch.
    :return: contentLoss
    '''
    relu22 = Relu_22()
    relu22.to('cuda' if torch.cuda.is_available() else 'cpu')

    output22_yc = relu22(y_c)
    output22_hat = relu22(y_hat)

    # flatten the tensors to make calculation easier and straightforward
    hat_flatten = output22_hat.reshape(output22_hat.shape[0], -1)
    yc_flatten = output22_yc.reshape(output22_yc.shape[0], -1)

    loss_content = criterion(hat_flatten, yc_flatten)
    return loss_content
