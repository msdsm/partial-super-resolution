import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from hat import PartialHAT
from dataset import HAT_Dataset_ImageNet
from tqdm import tqdm
import glob
import wandb
import random

#################### config ####################
DATA_PROCESSING = 'pixelization'
# DATA_PROCESSING = 'blur'
# DATA_PROCESSING = 'noise'
train_data_path = "/mnt/data/dataset/ImageNet/train/"
wandb.init(project="PROJECT-NAME")
alpha = 0.5 # [0.0, 1.0]
CHECKPOINT_PATH = "./checkpoint/CHECKPOINT-NAME.pth"
epochs = 100
batch_size = 8
learning_rate = 1e-4
IMG_SIZE = 64
#################### config ####################

os.makedirs("./checkpoint", exist_ok=True)
train_img_files = []
for file in glob.glob(train_data_path+"**", recursive=True):
    if os.path.isfile(file) and file.split("/")[-1].split(".")[-1] == "JPEG":
        train_img_files.append(file)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
])
train_dataset = HAT_Dataset_ImageNet(
    img_files = train_img_files,
    transform = transform,
    processing = DATA_PROCESSING,
    img_size = IMG_SIZE
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = PartialHAT(
    img_size=IMG_SIZE,
    upscale=1,
    upsampler='pixelshuffle',
    window_size=16
)
model.to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
for epoch in range(epochs):
    print("epoch {}:".format(str(epoch+1)))
    model.to(device)
    model.train()
    with tqdm(train_dataloader, total=len(train_dataloader)) as pbar:
        for i, (lowimgs, maskimgs, gtimgs) in enumerate(pbar):
            lowimgs = lowimgs.to(device)
            maskimgs = maskimgs.to(device)
            gtimgs = gtimgs.to(device)
            lowimgs = lowimgs.to(torch.float32)
            maskimgs = maskimgs.to(torch.float32)
            gtimgs = gtimgs.to(torch.float32)
            optimizer.zero_grad()
            output = model(lowimgs, maskimgs)
            batch_size = maskimgs.size(0)
            losses = []
            for b in range(batch_size):
                mask = maskimgs[b, 0]
                white_pixels = (mask == 1).nonzero(as_tuple=True)
                h1, h2 = white_pixels[0].min().item(), white_pixels[0].max().item() + 1
                w1, w2 = white_pixels[1].min().item(), white_pixels[1].max().item() + 1
                output_crop = output[b, :, h1:h2, w1:w2]
                gtimgs_crop = gtimgs[b, :, h1:h2, w1:w2]
                loss = alpha * criterion(output_crop, gtimgs_crop) + (1 - alpha) * criterion(output, gtimgs)
                losses.append(loss)
            batch_loss = torch.stack(losses).mean()
            wandb.log({"loss": batch_loss.item()})
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.detach().item())
    
    print("mean_train_loss:{}".format(sum(train_losses)/len(train_losses)))
    model.to("cpu")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print("save model")