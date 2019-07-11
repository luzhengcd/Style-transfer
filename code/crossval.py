import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import timing
from torchvision import models
from transferNet import TransferNet
from utils import *
import torch.nn as nn
from datasets import readData, styleImg
import lossCalculation
import glob


para_combo = [(8, 5e5), (3, 5e5), (20, 5e5),
              (8, 5e4), (8, 1e6)]



vgg16 = models.vgg16(pretrained=True)
vgg16.to('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


PATH_TRAIN_FILE = r'../data/train2014/*'
PATH_STYLE = r'../data/styleImg/' +'style_vangogh.jpg'
pic_path = glob.glob(PATH_TRAIN_FILE)
pic_path = pic_path[:8000]

NUM_PIC = len(pic_path)

NUM_EPOCHS = 2
BATCH_SIZE = 4
USE_CUDA = True
NUM_WORKERS = 0

y_s = styleImg(PATH_STYLE).to('cuda' if torch.cuda.is_available() else 'cpu')

model = TransferNet()

criterion = nn.MSELoss()

# criterion = lossCalculation.TotalLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

model.to(device)
criterion.to(device)

num_pic_each = 2000
torch.manual_seed(42)
torch.cuda.manual_seed(42)


for i in range(int(NUM_PIC / num_pic_each)):

# Streaming input the data
# Every 100(or 1000, depends) pictures will be loaded into memory, after finishing training the model
# with the 100 pictures, load the next 100 picture, and repeat till all the pictures
# have been used to train the network

    path_lst = pic_path[i*num_pic_each : (i+1)*num_pic_each]
    train_dataset = readData(path_lst)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)

    # train_losses = []

    for epoch in range(NUM_EPOCHS):
        print(epoch)
        train_loss = train(model, device, train_loader, optimizer, NUM_EPOCHS, y_s,
                           criterion, args.cWeight, args.sWeight)

    print( '===========[{0}/{1}]============\t'
           .format(i, int(NUM_PIC/num_pic_each)))


torch.save(model.state_dict(), '../model/' + args.outpath + '.pth')