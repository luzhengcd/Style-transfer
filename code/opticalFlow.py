import cv2
import numpy as np
import torch

def reshape_batch(batch, batch_size):
    res = batch.reshape((batch_size, *batch.shape[2:], batch.shape[1]))
    return res


def calFlow(pre_batch, current_batch):
    # the shape of the batch data would be (#frame, channel, height, weight)

    # the input tensor has the shape of (C, H, W)
    # but the default img shape in cv2 is (H, W, C)
    # so before calculating the flow, have to reshape the tensor a bit so that the two match
    batch_size = pre_batch.shape[0]
    pre_batch = reshape_batch(pre_batch, batch_size)
    current_batch = reshape_batch(current_batch, batch_size)

    flow_lst = [cv2.calcOpticalFlowFarneback(cv2.cvtColor(np.float32(pre_batch[i]),
                                                                       cv2.COLOR_BGR2GRAY),
                                                          cv2.cvtColor(np.float32(current_batch[i]),
                                                                       cv2.COLOR_BGR2GRAY),
                                                          None, 0.5, 3, 15, 3, 5, 1.2, 0)
                for i in range(batch_size)]

    return flow_lst


def warp_flow(batch, flow_lst):
    # note that img here is also a batch, which contains 2 images with the format of (#frame, C, H, W)
    # after reshape, the shape of the batch becomes (#frame, H, W, C), and the shape of flow tensor is (#frame, H, W, grad)

    batch_size = batch.shape[0]
    # batch size would be 2
    #
    # print(batch.shape)
    # print(batch_size)
    batch_reshape =reshape_batch(batch, batch_size)
    h, w = flow_lst[0].shape[:2]


    for i in range(batch_size):
        flow_lst[i][:, :, 0] += np.arange(h)
        flow_lst[i][:, :, 1] += np.arange(w)[:, np.newaxis]


    # cv2.remap(batch_reshape[0].data.numpy(), flow_lst[0], None, cv2.INTER_LINEAR)

    warped_lst = [torch.Tensor(cv2.remap(batch_reshape[i].data.numpy(), flow_lst[i],
                        None, cv2.INTER_LINEAR)) for i in range(batch_size)]

    warped = torch.stack(warped_lst)
    res = warped.reshape((batch_size, batch.shape[1], h, w))

    return res