
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from torchvision import models
from transferNet import TransferNet
from utils import *
from datasets import readData, styleImg
import lossCalculation
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-outpath', type = str, default='model')
args = parser.parse_args()


vgg16 = models.vgg16(pretrained=True)
vgg16.cuda()

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


PATH_TRAIN_FILE = r'../data/train2014/train2014/*'
PATH_STYLE = r'../data/style_vangogh.JPG'
pic_path = glob.glob(PATH_TRAIN_FILE)

NUM_PIC = len(pic_path)

NUM_EPOCHS = 2
BATCH_SIZE = 4
USE_CUDA = True
NUM_WORKERS = 0

y_s = styleImg(PATH_STYLE).cuda()

model = TransferNet()

criterion = lossCalculation.TotalLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

model.to(device)
criterion.to(device)

num_pic_each = 1000
print('total number of iteration: ', int(NUM_PIC / num_pic_each))

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
        train_loss = train(model, device, train_loader, optimizer, NUM_EPOCHS, y_s, criterion)
        # train_losses.append(train_loss)

    print( '===========[{0}/{1}]============\t'
           .format(i, int(NUM_PIC/num_pic_each)))


torch.save(model.state_dict(), '../model/' + args.outpath + '.pth')