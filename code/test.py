import glob
import IO
import time
import torch
from utils import findCutoff


if torch.cuda.is_available():
    TRAIN_LST = glob.glob(r'../../data/FlyingThings3D_subset/train/image_clean/left/*')
    PATH_FLOW = glob.glob(r'../../FlyingThings3D_subset/train/flow/left/into_past/*')
    PATH_OCCLUSION = glob.glob(r'../../data/FlyingThings3D_subset/train/flow_occlusions/left/into_past/*')
else:
    TRAIN_LST = glob.glob(r'../data/flow_data/image_clean_left/*')
    PATH_FLOW = glob.glob(r'../data/flow_data/flow_left/into_past/*')
    PATH_OCCLUSION = glob.glob(r'../data/flow_data/flow_occlusions/left/into_past/*')


TRAIN_LST.sort()
PATH_FLOW.sort()
PATH_OCCLUSION.sort()


# cutoff = findCutoff(PATH_FLOW)


# print(PATH_FLOW)
total_count = 0
error_count = 0

for i in PATH_FLOW:
    total_count += 1

    f = open(i, 'rb')
    header = f.read(4)

    if header.decode("utf-8") != 'PIEH':
        # print(e)
        error_count += 1
        print(i)
        print(error_count)

print('======= Total count: '+ str(total_count) + '=========')
print('======= Error count: '+ str(error_count) + '=========')


