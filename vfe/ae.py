import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from math import log10
import argparse
import sys
# Torch related libs
import torch
import torchvision
import torch.nn  as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose, CenterCrop, ToPILImage, Normalize
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# Custom import
from data import Oulu_Dataset
from utils import denormalize

"""Fully-connected Autoencoder VFE"""

class Encoder(nn.Module):

    def __init__(self, width, height):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(width * height, 1000)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 50)

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))
        x3 = self.lin3(x2)
        return x3


class Decoder(nn.Module):

    def __init__(self, width, height):
        super(Decoder, self).__init__()
        self.lin4 = nn.Linear(50, 500)
        self.lin5 = nn.Linear(500, 1000)
        self.lin6 = nn.Linear(1000, width * height)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x4 = self.relu(self.lin4(x))
        x5 = self.relu(self.lin5(x4))
        x6 = self.lin6(x5)
        x7 = self.tanh(x6)
        return x7 


class VFE(nn.Module):
   
    def __init__(self, width, height):
        super(VFE, self).__init__()
        self.encoder = Encoder(width, height)
        self.decoder = Decoder(width, height)

    def forward(self, x):
        self.lant_vec = self.encoder(x)
        res = self.decoder(self.lant_vec)
        return res

        
"""Convolutional Autoencoder VFE"""
class Conv_Encoder(nn.Module):

    def __init__(self, width, height):
        super(Conv_Encoder, self).__init__()
        self.lin1 = nn.Linear(width * height, 1000)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 50)

    def forward(self, x):
        x1 = self.relu(self.lin1(x))
        x2 = self.relu(self.lin2(x1))
        #x3 = self.lin3(x2)
        x3 = self.relu(self.lin3(x2))
        return x3


class Conv_Decoder(nn.Module):

    def __init__(self, width, height):
        super(Conv_Decoder, self).__init__()
        self.lin4 = nn.Linear(50, 500)
        self.lin5 = nn.Linear(500, 1000)
        self.lin6 = nn.Linear(1000, width * height)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x4 = self.relu(self.lin4(x))
        x5 = self.relu(self.lin5(x4))
        x6 = self.lin6(x5)
        x7 = self.tanh(x6)
        return x7 


class Conv_VFE(nn.Module):
   
    def __init__(self, width, height):
        super(Conv_VFE, self).__init__()
        self.encoder = Conv_Encoder(width, height)
        self.decoder = Conv_Decoder(width, height)

    def forward(self, x):
        self.lant_vec = self.encoder(x)
        res = self.decoder(self.lant_vec)
        return res


#"""
print("E.g python ae.py --view 1 --images_dir ext_frames1")
parser = argparse.ArgumentParser()
parser.add_argument("--view", type=int, required=True, help="View mode : type int 1-5")
parser.add_argument("--images_dir", type=str, required=True, help="Path to images")
opt = parser.parse_args()
print(opt)

if opt.view == 1:
    width = 53
    height = 31
elif opt.view == 2:
    width = 45
    height = 30
elif opt.view == 3:
    width = 45
    height = 31
elif opt.view == 4:
    width = 29
    height = 37
elif opt.view == 5:
    width = 24
    height = 33
else :
    sys.exit(1)

cudnn.benchmark = True 
####### Default config variables #############3
num_epochs = 27
BATCH_SIZE = 128
learning_rate = 1e-3
root_dir = "ext_frames" + str(opt.view)
##### Transforms ####
to_tens = ToTensor()
to_pil = ToPILImage()
normal = Normalize([0.5], [0.5])
scale = Resize((height, width))
gray = Grayscale(num_output_channels=1)
to_tensor = ToTensor()
composed = Compose([scale, gray, to_tensor, normal])

##### Data initialization #######
train_dt = Oulu_Dataset(root_dir, transform=composed)
#print(len(train_dt))
train_loader = DataLoader(train_dt, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True)

test_dt = Oulu_Dataset(root_dir, mode="test", transform=composed)
#print(len(test_dt))
test_loader = DataLoader(test_dt, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle= False)
# Model initialization
vfe = VFE(width, height).cuda()
#print(vfe)    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    vfe.parameters(), lr=learning_rate, weight_decay=1e-5
)
#"""

def train(epoch, writer, output_dir):
    """
    Args:
        epoch: int

        writer: torch.utils.tensorboard.Summarywriter object

        output_dir : str
    """
    vfe.train()
    


    train_loss = 0
    if epoch == 0:
        for batch_idx , data in enumerate(train_loader):
            img  = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            #======= forward =======
            output = vfe(img)
            #print(vfe.lant_vec.size())
            loss = criterion(output, img)
            train_loss += loss
            optimizer.zero_grad()

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('train_loss', train_loss, epoch)
        return

    for batch_idx , data in enumerate(train_loader):
        img  = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        #======= forward =======
        output = vfe(img)
        sample = output[:5]
        sample = sample.view(sample.size(0), 1,  height, width)
        loss = criterion(output, img)
        train_loss += loss
        #====== backward =======
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 and batch_idx < 20:
            writer.add_image('train_sample', vutils.make_grid(sample.data, normalize=True, scale_each=True), batch_idx)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    print(f"\nTrain set: Average loss @ epoch = {epoch} => {train_loss:.8f}")
    writer.add_scalar('train_loss', train_loss, epoch)
    # do checkpointing
    torch.save(vfe.state_dict(),"%s/vd_epoch_%d.pth" %(output_dir, epoch))

    

def test(epoch, writer):
    vfe.eval()
    test_loss = 0
    avg_psnr = 0
    for batch_idx, data in enumerate(test_loader):
        img = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        #======= forward =======
        output = vfe(img)
        sample = output[:5]
        sample = sample.view(sample.size(0), 1,  height, width)
        #print(sample.size())
        loss = criterion(output, img)
        psnr = 10 * log10(1 / loss.item())
        test_loss += loss
        #avg_psnr += psnr

        if batch_idx == 1 :
            orig_img = img[5]
            orig_img = orig_img.view(height, width)
            #print(orig_img.size())
            out_img = output[5]
            out_img = out_img.view(height, width)
            #print(out_img.size())
            diff_img = torch.abs(orig_img - out_img)
            diff_sclr = torch.sum(diff_img)
            orig_fl = "orig_img" + "_" + str(epoch) + "_" + str(opt.view) + ".png"
            out_fl = "out_img" + "_" + str(epoch) + "_" +  str(opt.view) + ".png"
            diff_fl = "diff_img" + "_" + str(epoch) + "_" +  str(opt.view) + ".png"
            folder = "tr_imgs/"
            save_image(orig_img,folder + orig_fl)
            save_image(out_img,folder + out_fl)
            save_image(diff_img, folder + diff_fl)
            writer.add_scalar('diff_sclr', diff_sclr, epoch)
    test_loss /= len(test_loader.dataset)
    avg_psnr = 10 * log10( 1 / test_loss.item())
    print(f"\nTest set: Average loss @ epoch = {epoch} => {test_loss:.8f}\n")
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_avg_psnr', avg_psnr, epoch)
    writer.add_image('sample', vutils.make_grid(sample.data, normalize=True, scale_each=True), epoch)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))



if __name__ == "__main__":
    # Writer will output to ./runs/ directory by default
    print(f"opt.view is :\t{opt.view}")
    output_dir = "out" + str(opt.view)
    print(output_dir) 
    runs_dir =  "runs" + str(opt.view)
    print(runs_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    if os.path.isdir(runs_dir):
        shutil.rmtree(runs_dir)

    writer = SummaryWriter(runs_dir)
    #output_dir = "out"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        pass

    for epoch in range(num_epochs):
        train(epoch, writer, output_dir)
        test(epoch, writer)

    writer.close()
