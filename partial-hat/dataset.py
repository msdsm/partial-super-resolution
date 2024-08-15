import torch
import torchvision
import glob
import random
import cv2
from PIL import Image
from utils.random_pixelization import apply_random_pixelization
from utils.random_blur import apply_random_blur
from utils.random_noise import apply_random_noise

class HAT_Dataset_ImageNet(torch.utils.data.Dataset):
    def __init__(self, img_files, transform, processing, img_size):
        self.img_files = img_files
        self.transform = transform
        self.processing = processing
        self.img_size = img_size
    
    def __getitem__(self, index):
        name = self.img_files[index].split("/")[-1]
        gtimg = cv2.imread(self.img_files[index])
        gtimg = cv2.resize(gtimg, (self.img_size, self.img_size))

        if self.processing == 'pixelization':
            lowimg, maskimg = apply_random_pixelization(gtimg, 8, 4)
            gtimg = Image.fromarray(gtimg)
            maskimg = Image.fromarray(maskimg).convert('L')
            lowimg = Image.fromarray(lowimg)
        elif self.processing == 'blur':
            lowimg, maskimg = apply_random_blur(gtimg, 8, 5)
            gtimg = Image.fromarray(gtimg)
            maskimg = Image.fromarray(maskimg).convert('L')
            lowimg = Image.fromarray(lowimg)
        else:
            lowimg, maskimg = apply_random_noise(gtimg, 8, 20)
            gtimg = Image.fromarray(gtimg)
            maskimg = Image.fromarray(maskimg).convert('L')
            lowimg = Image.fromarray(lowimg)
        lowimg = self.transform(lowimg)
        maskimg = self.transform(maskimg)
        gtimg = self.transform(gtimg)
        return (lowimg, maskimg, gtimg)

    def __len__(self):
        return len(self.img_files)