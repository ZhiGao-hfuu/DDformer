import numpy as np
import torch
from math import log10
import torch,torchvision.transforms as transforms
import torch.nn as nn
def PSNR(img_a,img_b,get_mse=False):
    ave_psnr,ave_mse=0,0
    for img1,img2 in zip(img_a,img_b):
        img1=torch.unsqueeze(img1,dim=0)
        img2=torch.unsqueeze(img2,dim=0)
        criterionMSE = nn.MSELoss()
        mse=criterionMSE(img1,img2)
        if mse.item() == 0:
            ave_psnr+= 100
        psnr = 10 * log10(1 / mse.item())
        if get_mse:
            ave_mse+=mse.item()
        ave_psnr+= psnr
    ave_psnr/=img_a.size(0)
    if get_mse:
        ave_mse/=img_a.size(0)
        return ave_psnr,ave_mse
    return ave_psnr

def val_log(net,val_data_loader,device,data_vis=None):
    net.eval()
    ave_psnr=0
    for iter, batch in enumerate(val_data_loader):
        input, target = batch[0].to(device), batch[1].to(device)
        restord_image = net(input).data
        restord_image = restord_image * 0.5 + 0.5
        target = target * 0.5 + 0.5
        psnr_value = PSNR(restord_image, target)
        ave_psnr += psnr_value
    ave_psnr /= len(val_data_loader)
    return ave_psnr

def one_image_from_GPU_tensor(tensor,tfs=True):
    """Scales a N*CxHxW tensor with values in the range [-1, 1] to [0, 255]"""
    image = tensor.cpu()
    one_image = image[0,:,:,:]
    one_image = torch.squeeze(one_image,dim=0)
    if tfs:
        one_image = 0.5 * one_image + 0.5  # [-1, 1] --> [0, 1]
    one_image = transforms.ToPILImage()(one_image)  # [0, 1] --> [0, 255]
    return one_image
