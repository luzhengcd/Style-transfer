import cv2
import numpy as np

def calFlow(pre, current):
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pre_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def warp_flow(img, flow):

    h, w = flow.shape[:2]
    # flow = - flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2. remap(img, flow, None, cv2.INTER_LINEAR)

    return res