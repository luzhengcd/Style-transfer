import scipy.misc as mic
import numpy as np
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
import lossCalculation
from PIL import Image
import os


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum  += val * n
        self.count += n
        self.avg = self.sum / self.count





def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    mic.imsave(path, img)



def Gram(X):

    # X would be a matrix with shape [#pic, ..., ..., ...] tensor
    # The output should have a shape of [#pic, dim, dim], which is [#pic, 64, 64] in our case
    # [#pic, channel, w, h] where w = h
    # the output should be [#pic, channel, channel]
    temp = X.shape
    X = X.reshape(temp[0], temp[1], temp[2] ** 2)
    num_pic = temp[0]

    gram_lst = []

    for num in range(num_pic):
        X_new = X[num]
        gram_temp = torch.mm(X_new, X_new.transpose(0,1))

        gram = gram_temp / np.prod(X_new.shape)
        gram_lst.append(gram)
    gram_tensor = torch.stack(gram_lst)
    return gram_tensor


def train(model, device, data_loader,
          optimizer, epoch, y_s, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()

    for i, input in enumerate(data_loader):
        data_time.update((time.time() - end))

        y_org = input[0]
        y_c = y_org

        input = y_org.to(device)

        optimizer.zero_grad()


        y_hat = model(input)

        loss = criterion(y_c, y_s, y_hat)
        loss.to(device)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses))

    return losses.avg

