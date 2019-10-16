import os
import functools
import matplotlib.pyplot as plt
import numpy as np
import shutil
import seaborn as sns
import pandas as pd
# Torch related libs
import torch
import torchvision
import torch.nn  as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose, CenterCrop, ToPILImage, Normalize
from torchvision.utils import save_image

#Normalization parameters for pytorch models
mean = np.array([0.5])
std = np.array([0.5])

def denormalize(tensors):
    """Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def plot_image(img_arr):
    """
    Args:
        tensor: pytorch tensor to cpu
    """
    #pil_img = to_pil(tensor)
    #img_arr = np.array(pil_img)
    if len(img_arr.shape) == 2:
        plt.gray()
    plt.imshow(img_arr)
    plt.show()


def file_is_image(filename):
    """Check if file is image.
    Args:
        filename: str
    Returns:
        bool
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def find_angle(filename):
    """Find type of view from file.
    Args:
        filename: str
    Returns:
        angle: int
    """
    return int(list(filename.split("_")[1])[1]) # "s2_v2_u32005.png" ==> v2 ==> 2


def list_files(root_dir):
    """
    Args:
        root_dir: (str) relative path to folder
    Returns:
         list of str
    """
    return os.listdir(root_dir)




def plot_from_csv(csv_file):
    """
    Args:
        csv_file : str
    """
    #plt.style.use('ggplot')
    f_pref = csv_file.split(".")[0]
    sns.set_style('whitegrid')
    df = pd.read_csv(csv_file)

    vals = df["Value"].values
    length = len(vals)

    if "psnr" in f_pref:
        plt.ylabel("Test PSNR")
    elif "train" in f_pref:
        plt.ylabel("Train loss")
    elif "test" in f_pref:
        plt.ylabel("Test loss ")
    elif "diff" in f_pref:
        plt.ylabel("Image diff")
    else:
        pass

    if "psnr" in f_pref:
        plt.title("PSNR (dB)")
    elif "diff" in f_pref:
        plt.title("Pixel diff")
    else:    
        plt.title("MSELoss")
    plt.xlabel("Epoch")
    #plt.xticks(range(0,length))
    plt.plot(range(0,length), vals)
    #plt.show()
    plt.savefig("../extras/"+f_pref+".png")
    plt.close()


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 31, 52)
    return x



if __name__ == "__main__":
    [plot_from_csv("test_loss_" + str(i) + ".csv" ) for i in range(1,6) ] 
    [plot_from_csv("train_loss_" + str(i) + ".csv") for i in range(1,6) ]
    [plot_from_csv("test_avg_psnr_"  + str(i) + ".csv") for i in range(1,6)]
    [plot_from_csv("diff_sclr_"  + str(i) + ".csv") for i in range(1,6)]