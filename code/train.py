
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torchvision import models
from transferNet import TransferNet
from utils import *
from datasets import readData, styleImg
import lossCalculation

parser = argparse.ArgumentParser()
parser.add_argument('-outpath', type = str, default='model')

vgg16 = models.vgg16(pretrained=True)
vgg16.cuda()

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


PATH_TRAIN_FILE = r'../data/testPic/*'
PATH_STYLE = r'../data/style_vangogh.JPG'
PATH_OUT = r'../model/firstModel.pth'

NUM_EPOCHS = 2
BATCH_SIZE = 4
USE_CUDA = True
NUM_WORKERS = 0
train_dataset = readData(PATH_TRAIN_FILE)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)
y_s = styleImg(PATH_STYLE).cuda()

model = TransferNet()

criterion = lossCalculation.TotalLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

model.to(device)
criterion.to(device)
train_losses = []


for epoch in range(NUM_EPOCHS):
    print(epoch)
    train_loss = train(model, device, train_loader, optimizer, NUM_EPOCHS, y_s, criterion)
    train_losses.append(train_loss)


torch.save(model.state_dict(), '../model/' + parser.outpath + '.pth')