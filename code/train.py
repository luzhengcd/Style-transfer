
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import timing
import torch
from transferNet import TransferNet
from utils import *
import torch.nn as nn
from datasets import readData, styleImg, readVideo
import lossCalculation
import glob
from utils import AverageMeter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-outpath', type = str, default='modelVideo1')
parser.add_argument('-style', type = str, default= 'style_vangogh')
parser.add_argument('-cWeight', type = float, default = 8)
parser.add_argument('-sWeight', type = float, default= 500000)
parser.add_argument('-trainSize', type = int, default=20)
parser.add_argument('-oWeight', type = float, default = 100)
# parser.add_argument('-fWeight', type = float, default=1000)
args = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

vgg16 = models.vgg16(pretrained=True)
vgg16.to('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


PATH_TRAIN_FILE = r'../data/video/*'
PATH_STYLE = r'../data/styleImg/'+ args.style +'.jpg'
pic_path = glob.glob(PATH_TRAIN_FILE)
pic_path = pic_path[:args.trainSize]

NUM_VIDEO = len(pic_path)



val_size = int(NUM_VIDEO * 0.1)

VAL_lst = np.random.choice(pic_path, val_size)
TRAIN_lst = list(set(pic_path) - set(VAL_lst))



NUM_EPOCHS = 2
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
for i in range(len(TRAIN_lst)):
    torch.cuda.empty_cache()
    # print('memory management: ', torch.cuda.memory_allocated(device))

# Streaming input the data
# Every 100(or 1000, depends) pictures will be loaded into memory, after finishing training the model
# with the 100 pictures, load the next 100 picture, and repeat till all the pictures
# have been used to train the network

    # path_lst = pic_path[i*num_pic_each : (i+1)*num_pic_each]
    train_dataset = readVideo(TRAIN_lst[i])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False, num_workers=NUM_WORKERS)

    # train_losses = []
    best_val_loss = 0.0


    for epoch in range(NUM_EPOCHS):
        print(epoch)

        train_loss = train(model, device, train_loader, optimizer, NUM_EPOCHS, y_s,
                           criterion, args.cWeight, args.sWeight, args.oWeight, epoch)

    losses = AverageMeter()
    loss_content_track = AverageMeter()
    loss_style_track = AverageMeter()
    loss_temporal_track = AverageMeter()

    model.eval()
    for p in range(val_size):
        torch.cuda.empty_cache()

        # change name to train dataset and train loader to avoid the memory error
        val_dataset = readVideo(VAL_lst[p])

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=NUM_WORKERS)

        loss_avg, sLoss_avg, cLoss_avg, tLoss_avg = evaluate(model, device, val_loader, criterion,
                                                             args.sWeight, args.cWeight, y_s, args.oWeight)

    loss_temporal_track.update(tLoss_avg)
    loss_content_track.update(cLoss_avg)
    loss_style_track.update(sLoss_avg)
    losses.update(loss_avg)

    is_best = losses.avg < best_val_loss

    if is_best:
        torch.save(model.state_dict(), '../model/' + args.outpath + '.pth')

    print( '===========[{0}/{1}]============\t'
           .format(i, NUM_VIDEO))


print('===== Finished =====')


