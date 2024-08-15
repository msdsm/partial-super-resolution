import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from drct import PartialDRCT
from dataset import DRCT_Dataset_ImageNet
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
from torchvision.utils import save_image
import random
import glob


#################### config ####################
LOAD_PATH = "./checkpoint/CHECKPOINT-NAME.pth"
SAVE_PATH = "./result/imagenet/SAVE-PATH-NAME/"
IMG_SIZE = 64 
DATA_PROCESSING = 'pixelization'
# DATA_PROCESSING = 'blur'
# DATA_PROCESSING = 'noise'
test_data_path = "/mnt/data/dataset/ImageNet/val/"
#################### config ####################


os.makedirs(SAVE_PATH, exist_ok=True)
test_img_files = []
for file in glob.glob(test_data_path+"**", recursive=True):
    if os.path.isfile(file) and file.split("/")[-1].split(".")[-1] == "JPEG":
        test_img_files.append(file)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
])
test_dataset = DRCT_Dataset_ImageNet(
    img_files = test_img_files,
    transform = transform,
    processing = DATA_PROCESSING,
    img_size = IMG_SIZE
)
test_dataloader = DataLoader(test_dataset, batch_size=1)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = PartialDRCT(
    img_size=IMG_SIZE,
    upscale=1,
    upsampler='pixelshuffle',
    window_size=16
)  
model.to(device)
parameters_load_path = LOAD_PATH
model.load_state_dict(torch.load(parameters_load_path))
model.eval()
cnt = 0
psnr_sum = 0.0
with tqdm(test_dataloader, total=len(test_dataloader)) as pbar:
    for i, (lowimg, maskimg, gtimg) in enumerate(pbar):
        lowimg = lowimg.to(device)
        gtimg = gtimg.to(device)
        maskimg = maskimg.to(device)
        output = model(lowimg, maskimg)
        output = output.squeeze(0)
        output_img = output.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
        output_img = np.clip(output_img*255.0, a_min=0, a_max=255).astype(np.uint8)
        gtimg = gtimg.squeeze(0)
        gtimg = gtimg.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
        gtimg = np.clip(gtimg*255.0, a_min=0, a_max=255).astype(np.uint8)
        cnt += 1
        maskimg = maskimg.squeeze(0)
        maskimg = maskimg.to("cpu").detach().numpy().copy().transpose(1, 2, 0).astype(np.float32)
        maskimg = np.clip(maskimg*255.0, a_min=0, a_max=255).astype(np.uint8)
        mask = maskimg.squeeze(axis=-1) # (h, w, 1) -> (h, w)
        output_img[mask == 0] = gtimg[mask == 0]
        psnr_sum += cv2.PSNR(output_img, gtimg)
        output_img = Image.fromarray(output_img)
        output_img.save(SAVE_PATH + str(i) + '.jpg')

psnr = psnr_sum / cnt
print(psnr)