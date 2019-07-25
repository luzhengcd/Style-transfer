import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import timing
from torchvision import models
from transferNet import TransferNet
from utils import *
import torch.nn as nn
from datasets import readData, styleImg
import glob


'''
This script puts everything together, and trains a style transfer model.

It includes the following parts:

    - make the training and validation datasets using datasets.py
    - initialize a network and set parameters based
    - train the model by updating the parameters with back propagation
    - evaluate the model on validation datasets
    - save the model which has smallest validation loss
    
'''


parser = argparse.ArgumentParser()
parser.add_argument('-outpath', type = str, default='model')
parser.add_argument('-style', type = str, default= 'mosaic')
parser.add_argument('-cWeight', type = float, default=1)
parser.add_argument('-sWeight', type = float, default= 2000)
parser.add_argument('-trainSize', type = int, default=2000)
args = parser.parse_args()

vgg16 = models.vgg16(pretrained=True)
vgg16.to('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

# specify the path to train data and style image
PATH_TRAIN_FILE = r'../data/train2014/*'
PATH_STYLE = r'../data/styleImg/'+ args.style +'.jpg'
pic_path = glob.glob(PATH_TRAIN_FILE)
pic_path = pic_path[:args.trainSize]

NUM_PIC = len(pic_path)

# validation dataset account for 10% of the full dataset
val_size = int(NUM_PIC * 0.1)

# set the parameters
NUM_EPOCHS = 2
BATCH_SIZE = 4
USE_CUDA = True
NUM_WORKERS = 0

# split out the validation dataset
VAL_lst = np.random.choice(pic_path, val_size)
TRAIN_lst = list(set(pic_path) - set(VAL_lst))

VAL_dataset = readData(VAL_lst)
VAL_loader = torch.utils.data.DataLoader(VAL_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)
# read in style image
y_s = styleImg(PATH_STYLE).to('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the network
model = TransferNet()

# choose MSE Loss as our loss function
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr = 0.001)

model.to(device)
criterion.to(device)

# To avoid the memory error, we read in 2000 pictures each time.

num_pic_each = 2000
torch.manual_seed(42)
torch.cuda.manual_seed(42)

for i in range(int(NUM_PIC / num_pic_each)):

    path_lst = TRAIN_lst[i*num_pic_each : (i+1)*num_pic_each]
    train_dataset = readData(path_lst)

    # make the train dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=NUM_WORKERS)

    best_val_loss = 0.0

    # loop over the number of epoch, train and evaluate the model.
    for epoch in range(NUM_EPOCHS):
        print(epoch)

        train_loss = train(model, device, train_loader, optimizer, NUM_EPOCHS, y_s,
                           criterion, args.cWeight, args.sWeight)

        val_losses, val_styleLoss, val_contentLoss = evaluate(model, device,VAL_loader, criterion,
                                                              args.sWeight, args.cWeight, y_s)

        is_best = val_losses < best_val_loss
        if is_best:
            torch.save(model.state_dict(), '../model/' + args.outpath + '.pth')

    print( '===========[{0}/{1}]============\t'
           .format(i, int(NUM_PIC/num_pic_each)))


