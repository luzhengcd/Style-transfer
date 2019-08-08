import scipy.misc as mic
import time
from lossCalculation import *
import IO

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

# find the cutoff of video
def findCutoff(path_lst):

    im_index = np.array([int(i[-11:-4]) for i in path_lst])
    # im_index.sort()

    im_index_shift = np.roll(im_index, 1)
    # print(im_index)
    diff = im_index - im_index_shift
    cutoff = [i for i in range(len(diff)) if diff[i] >= 2]
    cutoff.insert(0, 0)
    # print(cutoff)
    return cutoff


def Gram(X):

    # X would be a matrix with shape [#pic, ..., ..., ...] tensor
    # The output should have a shape of [#pic, dim, dim], which is [#pic, 64, 64] in our case
    # [#pic, channel, w, h] where w = h
    # the output should be [#pic, channel, channel]
    temp = X.shape
    X = X.reshape(temp[0], temp[1], temp[2] * temp[3])
    num_pic = temp[0]

    gram_lst = []

    for num in range(num_pic):
        X_new = X[num]
        gram_temp = torch.mm(X_new, X_new.transpose(0,1))

        gram = gram_temp / np.prod(X_new.shape)
        gram_lst.append(gram)
    gram_tensor = torch.stack(gram_lst)
    return gram_tensor



def crop_array(h, w, img_arr):

    shape = img_arr.shape
    height = shape[0]
    width = shape[1]
    each_h = int((height - h) / 2)
    each_w = int((width - w) / 2)
    return img_arr[each_h : height - each_h, each_w : width - each_w]


def train(model, device, data_loader, occlusion_path, flow_path,
          optimizer, y_s, criterion, cWeight, sWeight,
          oWeight):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_content_track = AverageMeter()
    loss_style_track = AverageMeter()
    loss_temporal_track = AverageMeter()

    model.train()

    end = time.time()

    data_iterator = enumerate(data_loader)

    # pre = next(data_iterator)[1][0]
    # print(pre.shape)
    flow_occlusion_flag = 0

    flow_length = len(flow_path)

    # print('length of flow: ', len(flow_path))
    for i, input in data_iterator:

        if flow_occlusion_flag + 1 > flow_length:

            continue

        current_flow_path = flow_path[flow_occlusion_flag]
        current_occlusion_path = occlusion_path[flow_occlusion_flag]

        # print('FLAG: ', flow_occlusion_flag)
        # print('Current occlusion: ', current_occlusion_path)
        # print('Current flow: ', current_flow_path)

        flow_occlusion_flag += 2

        # this is the input frame!!!!

        data_time.update((time.time() - end))
        y_c = input[0]
        num_frame = y_c.shape[0]

        if num_frame==1:
            continue

        current_input = y_c.to(device)

        optimizer.zero_grad()


        feature_current, y_hat_current = model(current_input)
        # feature_pre, y_hat_pre = model(pre_input)


        loss_content = cWeight * contentLoss(current_input, y_hat_current, criterion)
        loss_style = sWeight * styleLoss(y_s, y_hat_current, criterion)

        # def temporalF(f_current, f_pre, flow, criterion):
        # def temporalO(O_current, O_pre, I_current, I_pre, criterion, flow):

        # print('feature pre shape',feature_pre.shape)
        # print('feature current shape', feature_current.shape)
        # should have the same order
        # def temporalF(batch, flow_path, occlusion_path, criterion):

        loss_temporalF = oWeight * temporalF(current_input, current_flow_path, current_occlusion_path, criterion)
        # loss_temporalF = temporalF(feature_current, feature_pre, flow, criterion)
        if loss_temporalF == 0:
            continue

        loss = loss_content + loss_style  + loss_temporalF
        loss.to(device)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        loss_content_track.update(loss_content)
        loss_style_track.update(loss_style)
        loss_temporal_track.update(loss_temporalF)

        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Content loss {closs.val:.4f} ({closs.avg:.4f})\t'
              'Style loss {sloss.val:.4f} ({sloss.avg:.4f})\t'
              'Temporal loss {oloss.val:.4f} ({oloss.avg:.4f})'
              .format(i + 1, len(data_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                      closs = loss_content_track, sloss = loss_style_track, oloss = loss_temporal_track))

    return losses.avg

def evaluate(model, device, data_loader, criterion,  sWeight, cWeight, y_s, oWeight, print_freq = 10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_content_track = AverageMeter()
    loss_style_track = AverageMeter()
    loss_temporal_track = AverageMeter()


    end = time.time()

    data_iterator = enumerate(data_loader)

    pre = next(data_iterator)[1][0]

    for i, input in data_iterator:
        # this is the input frame!!!!
        # print('memory management: ', torch.cuda.memory_allocated(device))
        data_time.update((time.time() - end))

        y_c = input[0]

        current_input = y_c.to(device)
        pre_input = pre.to(device)

        feature_current, y_hat_current = model(current_input)
        feature_pre, y_hat_pre = model(pre_input)

        pre = current_input

        loss_content = cWeight * contentLoss(current_input, y_hat_current, criterion)
        loss_style = sWeight * styleLoss(y_s, y_hat_current, criterion)

        flow = calFlow(pre_input, current_input)


        loss_temporalO = oWeight * temporalO(y_hat_current, y_hat_pre, criterion, flow)
        # loss_temporalF = temporalF(feature_current, feature_pre, flow, criterion)

        loss = loss_content + loss_style + loss_temporalO

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        loss_content_track.update(loss_content)
        loss_style_track.update(loss_style)
        loss_temporal_track.update(loss_temporalO)

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Content loss {closs.val:.4f} ({closs.avg:.4f})\t'
              'Style loss {sloss.val:.4f} ({sloss.avg:.4f})\t'
              'Temporal loss {oloss.val:.4f} ({oloss.avg:.4f})'
              .format( i, len(data_loader), batch_time=batch_time, data_time=data_time,
                      loss=losses,
                      closs=loss_content_track, sloss=loss_style_track, oloss=loss_temporal_track))

    return losses.avg, loss_style_track.avg, loss_content_track.avg, loss_temporal_track.avg
