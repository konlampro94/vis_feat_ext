import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
from math import log10
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
import pandas as pd
import numpy as np
import argparse

# Custom import
from data import Oulu_Dataset, Extract_Dataset
from utils import denormalize
from ae import VFE
from utils import list_files
# kaldiio import
import kaldiio



def write_kaldi_matrix(folder, model, loader, output_dir):
    """
    Args:

        folder: str

        model: pytorch model

        loader: torch.utils.data.Dataloader

        output_dir: str
    """
    if  not os.path.exists(output_dir):
        print("Destination folder for storage\n of .ark , .scp  files doesn't exist!!")
        sys.exit(1)

    with torch.no_grad():
        for idx , img in enumerate(loader):
            img = img.view(img.size(0), -1).cuda()
            output = model(img)
            lt_vec = model.lant_vec.cpu()
            feats_np = lt_vec.numpy()
    # initially --> s1_v1_u32 , to --> s01_u32.ark to match audio name files 
    #print(folder)
    #s1 --> s01
    if folder.index("_") - folder.index("s") == 2:
        folder = "s0" + folder.split("s")[1]
    #s01_v1_u32 --> s01_u32
    tokens = folder.split("_")
    folder = tokens[0] + "_" + tokens[2]
    #print(folder)
    scp_file = output_dir + "/" + folder + ".scp"
    ark_file = output_dir + "/" + folder + ".ark"
    
    write_dict = {}
    write_dict [folder] = feats_np
    kaldiio.save_ark(ark_file, write_dict, scp=scp_file)



def main():
    print("E.g: python extract.py --spec_model vd_epoch_20.pth --models_dir out --view 1 --mode train")
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_model", type=str,  required=True , help="specific model")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to folder of checkpoint models!")
    parser.add_argument("--view", type=str , required=True, help="View angle for extract")
    parser.add_argument("--mode", type=str, required=True, help="Possible inputs: a) train b) test ")
    #print("hello")
    opt = parser.parse_args()
    print(opt)
    model_path = os.path.join(opt.models_dir, opt.spec_model)
    ##### Transforms ####
    #opt.view = str(opt.view)
    if int(opt.view) == 1:
        width = 53
        height = 31
    elif int(opt.view) == 2:
        width = 45
        height = 30
    elif int(opt.view) == 3:
        width = 45
        height = 31
    elif int(opt.view) == 4:
        width = 29
        height = 37
    elif int(opt.view) == 5:
        width = 24
        height = 33
    else :
        print("Wrong --view input.Please try again.")
        sys.exit(1)

    normal = Normalize([0.5], [0.5])
    scale = Resize((width, height))
    gray = Grayscale(num_output_channels=1)
    to_tensor = ToTensor()
    composed = Compose([scale, gray, to_tensor, normal])
    # Model initialization
    #vfe = VFE().cuda()
    vfe = VFE(width, height).cuda() # NEW CODE CHECK
    vfe.load_state_dict(torch.load(model_path))
    vfe.eval()
    view = opt.view
    mode = opt.mode  
    #view = "1"  ## opt.view

    folder = mode + view # train1, train2, test1, test2
    subfolders = os.listdir(folder)
    #print(subfolders)
    output_dir = opt.mode + "_video_" + view #
    print(f"Output dir : {output_dir}")
    #If folder exists delete and create new folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)

    fold_10 = subfolders#[:10]
    for i in range(len(fold_10)):
        dataset = Extract_Dataset(folder+"/"+fold_10[i], transform=composed)
        loader = DataLoader(dataset, batch_size=len(dataset))
        write_kaldi_matrix(fold_10[i], vfe, loader, output_dir)



if __name__ == "__main__":
    main()
    