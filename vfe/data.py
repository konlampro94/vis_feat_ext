import os
from PIL import Image
import torch
from skimage import io, transform

import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose, CenterCrop, ToPILImage


class Oulu_Dataset(Dataset):
    """Ouluvs2 Dataset! http://www.ee.oulu.fi/research/imag/OuluVS2/"""


    def __init__(self, root_dir, split_factor=0.8, mode="train", img_list=None, transform=None):
        """
        Args:
             root_dir : str
             img_labels : list of str
             full_img_labels : list of str
        """
        super(Oulu_Dataset, self).__init__()
        self.root_dir = root_dir
        #self.img_labels = img_list
        img_labels = os.listdir(root_dir)

        total_number = len(img_labels)
        splt_ind = int(split_factor * total_number)

        if mode == "train":    
            self.img_labels = img_labels[:splt_ind]
        else:
            self.img_labels = img_labels[splt_ind:]

        self.transform = transform
        self.full_img_labels =[ os.path.join(self.root_dir, img) for img in self.img_labels ]
        #self.img_list = [ np.array(Image.open(img)) for img in self.full_img_labels ]

    def __getitem__(self, index):
        """
        Args:
            index : int
        
        Returns:
            img : PIL Image
        """
        #img = self.img_list[index]
        img_label = self.full_img_labels[index]
        img = Image.open(img_label) 
        if self.transform:
            img = self.transform(img)
        return img

    def get_label(self, img):
        """
        Args:
            img: n-darray
        Returns:
            label : str
        """
        #idx = self.img_list.index(img)
        #label = self.img_labels[idx]
        pass
        #return label

    def __len__(self):
        return len(self.full_img_labels)


class Extract_Dataset(Dataset):

    def __init__(self, root_dir, img_list=None, transform=None):
        """
        Args:
             root_dir : str
             img_labels : list of str
             full_img_labels : list of str
        """
        super(Extract_Dataset, self).__init__()
        self.root_dir = root_dir
        #self.img_labels = img_list
        self.img_labels = os.listdir(root_dir)

        
        self.transform = transform
        self.full_img_labels =[ os.path.join(self.root_dir, img) for img in self.img_labels ]
        #self.img_list = [ np.array(Image.open(img)) for img in self.full_img_labels ]

    def __getitem__(self, index):
        """
        Args:
            index : int
        
        Returns:
            img : PIL Image
        """
        #img = self.img_list[index]
        img_label = self.full_img_labels[index]
        img = Image.open(img_label) 
        if self.transform:
            img = self.transform(img)
        return img


    def __len__(self):
        return len(self.full_img_labels)



def plot_image(img):
    """
    Args:
        img: PIL Image
    """
    #img = Image.open(filename)
    img = np.asarray(img)
    print(len(img.shape))
    if len(img.shape) == 2:
        plt.gray()
    print(f"numpy shape of image is:\t{img.shape}")
    plt.imshow(img)
    plt.show()

def transf_to_gray(img):
    gray_img = img.convert('L')
    gray_conv = Grayscale(num_output_channels=1)
    gray2_img = gray_conv(img)
    print(f"gray_img == gray2_img:\t{gray_img==gray2_img}")


if __name__ == "__main__":
    print("Hello from data.py")
    scale2 = Resize((31, 52))
    #scale = Resize((62, 106))
    gray = Grayscale(num_output_channels=1)
    #to_pil = ToPILImage()
    to_tensor = ToTensor()
    #cnt_crop = CenterCrop((31, 52))
    composed = Compose([scale2, gray, to_tensor])
    BATCH_SIZE = 64
    root_dir="ext_frames"
    img_files =  os.listdir(root_dir)
    print(int(0.8*len(img_files)))
    #print(f"Image files:\n{img_files[:10]}")
    #full_img_files = [os.path.join(root_dir, img) for img in img_files ]
    train_dt = Oulu_Dataset(root_dir, transform=composed)
    
    train_loader = DataLoader(train_dt, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

    test_dt = Oulu_Dataset(root_dir, mode="test", transform=composed)

    test_loader = DataLoader(test_dt, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)


     