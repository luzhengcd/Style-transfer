import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from transferNet import TransferNet
from utils import *
import torch.nn as nn
from datasets import readData, styleImg, readVideo
import glob
import IO
import timing
from utils import AverageMeter
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-modelIdx', type = int, default=0)
parser.add_argument('-style', type = str, default= 'style_vangogh')
parser.add_argument('-cWeight', type = float, default = 8)
parser.add_argument('-sWeight', type = float, default= 500000)
parser.add_argument('-trainset', type = str, default='val')
parser.add_argument('-oWeight', type = float, default = 100)
parser.add_argument('-testIdx', type = int, default = 0)

# parser.add_argument('-fWeight', type = float, default=1000)
args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

vgg16 = models.vgg16(pretrained=True)
vgg16.to('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    TRAIN_LST = glob.glob(r'../../data/FlyingThings3D_subset/' + args.trainset + '/image_clean/left/*')
    PATH_FLOW = glob.glob(r'../../FlyingThings3D_subset/' + args.trainset + '/flow/left/into_past/*')
    PATH_OCCLUSION = glob.glob(r'../../data/FlyingThings3D_subset/' + args.trainset + '/flow_occlusions/left/into_past/*')
else:
    TRAIN_LST = glob.glob(r'../data/flow_data/image_clean_left/*')
    PATH_FLOW = glob.glob(r'../data/flow_data/flow_left/into_past/*')
    PATH_OCCLUSION = glob.glob(r'../data/flow_data/flow_occlusions/left/into_past/*')

PATH_STYLE = r'../data/styleImg/'+ args.style +'.jpg'
# pic_path = glob.glob(PATH_TRAIN_FILE)

# empty flow count = 2641
# total flow count = 18711

# Remove empty flows

PATH_FLOW_EMPTY = []

for i in PATH_FLOW:
    try:
        IO.read(i)
    except:
        PATH_FLOW_EMPTY.append(i)


PATH_FLOW_VALID = list(set(PATH_FLOW) - set(PATH_FLOW_EMPTY))
print('==== Total number of valid flow ' + str(len(PATH_FLOW_VALID)))
TRAIN_LST.sort()
PATH_FLOW_VALID.sort()
PATH_OCCLUSION.sort()
# time.sleep(180)

cutoff = findCutoff(PATH_FLOW_VALID)

cutoff = cutoff[args.testIdx:]

# print(PATH_FLOW)
# print(cutoff)
# pic_path = pic_path[:args.trainSize]
#
# NUM_VIDEO = len(pic_path)
#
#
#
# val_size = int(NUM_VIDEO * 0.1)
#
# VAL_lst = np.random.choice(pic_path, val_size)
# TRAIN_lst = list(set(pic_path) - set(VAL_lst))

# error occurs when 1042
NUM_EPOCHS = 1
BATCH_SIZE = 2
USE_CUDA = True
NUM_WORKERS = 0

y_s = styleImg(PATH_STYLE).to('cuda' if torch.cuda.is_available() else 'cpu')

model = TransferNet()

criterion = nn.MSELoss()


optimizer = optim.Adam(model.parameters(), lr = 0.001)

model.to(device)
criterion.to(device)

# num_pic_each = 1000
# print('memory management: ', torch.cuda.memory_allocated(device))
start = 0

num_video = len(cutoff)

restart = True
count = 1

while restart:

    if count == 2:

        restart = False

    for i in range(1, num_video):

        print('====== Video [{0}/{1}] Iteration {2} ======'.format(i, num_video, count))

        torch.cuda.empty_cache()

        end = i

        idx_img_start = int(PATH_FLOW_VALID[cutoff[start]][-11 : -4]) - 1
        idx_img_end = int(PATH_FLOW_VALID[cutoff[end] - 1][-11 : -4]) + 1

        occlussion_path = PATH_OCCLUSION[cutoff[start]:cutoff[end]]

        flow_path = PATH_FLOW_VALID[cutoff[start] : cutoff[end]]

        temp_train_lst = TRAIN_LST[idx_img_start : idx_img_end]


        start = i
        train_dataset = readData(temp_train_lst)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               shuffle=False, num_workers=NUM_WORKERS)
        # train_losses = []
        # best_val_loss = 0.0


        # print(temp_train_lst)
        # print(occlussion_path)
        # print(flow_path)

        # for epoch in range(NUM_EPOCHS):
        #     print(epoch)


            # def train(model, device, data_loader, occlusion_path, flow_path,
            #           optimizer, epoch, y_s, criterion, cWeight, sWeight,
            #           oWeight, current_epoch):

        train_loss = train(model, device, train_loader, occlussion_path, flow_path,
                           optimizer, y_s, criterion, args.cWeight, args.sWeight, args.oWeight)
        #
        # losses = AverageMeter()
        # loss_content_track = AverageMeter()
        # loss_style_track = AverageMeter()
        # loss_temporal_track = AverageMeter()

        # model.eval()
        # for p in range(val_size):
        #     torch.cuda.empty_cache()
        #
        #     # change name to train dataset and train loader to avoid the memory error
        #     val_dataset = readVideo(VAL_lst[p])
        #
        #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
        #                                    shuffle=False, num_workers=NUM_WORKERS)
        #
        #     loss_avg, sLoss_avg, cLoss_avg, tLoss_avg = evaluate(model, device, val_loader, criterion,
        #                                                          args.sWeight, args.cWeight, y_s, args.oWeight)
        #
        # loss_temporal_track.update(tLoss_avg)
        # loss_content_track.update(cLoss_avg)
        # loss_style_track.update(sLoss_avg)
        # losses.update(loss_avg)

        # is_best = losses.avg < best_val_loss

        # if is_best:
        #     torch.save(model.state_dict(), '../model/' + args.outpath + '.pth')

        # print( '===========[{0}/{1}]============\t'
        #        .format(i, NUM_VIDEO))

    count += 1



torch.save(model.state_dict(), '../model/new_video_vangogh' + str(args.modelIdx) + '.pth')


print('===== Finished =====')


