import os
import random
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import torch
from networks import  define_D,MutiTransformer

from data_operation import DataLoaderTrain,DatasetFromFolder_in_test_mode
from base_option import BaseOptions
import time
from util import val_log

opt = BaseOptions().init().parse_args()
device=torch.device(opt.device)
epoch_start=0
# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

#create model
net_g=MutiTransformer(resolution=opt.resolution).to(device)
net_d=define_D(netd='mutiscale').to(device)

#leaning rate update
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch  - opt.niter) / float(opt.niter_decay + 1)
    return lr_l
optimizer_g=torch.optim.Adam(net_g.parameters(),lr=opt.lr,betas=(opt.beta,0.999))
optimizer_d=torch.optim.Adam(net_d.parameters(),lr=opt.lr,betas=(opt.beta,0.999))
scheduler_g=torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda_rule)
scheduler_d=torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lambda_rule)
loss_l1=nn.L1Loss().to(device)
#create dataset
train_dataset=DataLoaderTrain(opt.dataset+'trainA/',opt.dataset+'trainB/',resolution=opt.resolution,use_data_augmentation=True)
train_data_loader=DataLoader(dataset=train_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=2)
unpaired_dataset=DatasetFromFolder_in_test_mode(opt.unpaired_image)
unpaired_loader=DataLoader(dataset=unpaired_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=2)
val_dataset=DataLoaderTrain(opt.valset+'valA/',opt.valset+'valB/',resolution=opt.resolution,use_data_augmentation=False)
val_data_loader=DataLoader(dataset=val_dataset,batch_size=opt.batch_size,shuffle=False,num_workers=2)

if __name__ == '__main__':
    #tarin
    print('batch_size=',opt.batch_size,'device=',device,'data_num=',len(train_data_loader))
    #value parameter
    best_psnr = 0
    best_epoch=0
    least_loss = 2
    best_l1 = 1
    MD_iters = 3
    G2loss = 0
    opt.image_save_status = False
    resume=False
    if os.path.isfile(opt.save_model+'netG_latest.pth') and resume:
        checkpoint = torch.load(opt.save_model+'netG_latest.pth')
        epoch_start = checkpoint['epoch']+1
        net_g.load_state_dict(checkpoint['netG_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_g.load_state_dict(checkpoint['lr_learning_rate'])
        checkpoint = torch.load(opt.save_model + 'netD_latest.pth')
        net_d.load_state_dict(checkpoint['netD_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_d.load_state_dict(checkpoint['lr_learning_rate'])
        best_psnr = 0
        best_l1=1
        # epoch_start=0
    if not os.path.exists(opt.save_model):
        os.makedirs(opt.save_model)
    if opt.image_save_status and not os.path.exists(opt.save_image):
        os.makedirs(opt.save_image)

    for epoch in range(epoch_start,opt.epoch_end):
        print('epoch',epoch)
        ave_loss=0
        loss_d_log = 0
        l1_log,gan_log,laplace_log,unparied_log=0,0,0,0
        net_g.train()
        unpaired_iter = iter(unpaired_loader)

        t1=time.time()
        for iteration,batch in enumerate(train_data_loader):
            input,target=batch[0].to(device),batch[1].to(device)
            restord_image=net_g(input)
            ######################
            # (1) Update D network
            ######################
            for i in range(MD_iters):
                optimizer_d.zero_grad()
                pred_fake = net_d.forward(restord_image.detach())
                pred_real = net_d.forward(target)
                loss_d_fake = pred_fake.mean().unsqueeze(0)
                loss_d_real = pred_real.mean().unsqueeze(0)
                loss_d = (-loss_d_real + loss_d_fake)*0.5
                loss_d_log += loss_d.item()/3
                loss_d=torch.clip(loss_d,-1,1)
                loss_d.backward()
                optimizer_d.step()
            ######################
            # (2) unpaired Update G network
            ######################
            optimizer_g.zero_grad()
            # First, G(A) should fake the discriminator
            if iteration%3==0:
                try:
                    unpaired=unpaired_iter.__next__().to(device)
                except:
                    unpaired_iter = iter(unpaired_loader)
                    unpaired = unpaired_iter.__next__().to(device)
                unpaired_image = net_g(unpaired)
                pred_fake = net_d.forward(unpaired_image)
                loss_g_unpaired = pred_fake.mean().unsqueeze(0) * 0.5
                G2loss=loss_g_unpaired.item()
                unparied_log+=loss_g_unpaired.item()*3
                loss_g_unpaired.backward(retain_graph=True)
            ######################
            # (3) paired Update G network
            ######################
            # optimizer_g.zero_grad()
            pred_fake = net_d.forward(restord_image)
            loss_g_gan = pred_fake.mean().unsqueeze(0)
            loss_g_l1 = loss_l1(restord_image, target)
            loss_g = loss_g_gan*0.2+ loss_g_l1*10#+loss_g_ssim*2
            l1_log+=loss_g_l1.item()*10
            gan_log+=loss_g_gan.item()*0.2
            ave_loss += loss_g.item()
            loss_g.backward()
            optimizer_g.step()
        scheduler_g.step()
        scheduler_d.step()
        print(optimizer_g.param_groups[0]['lr'])
        ave_loss /= (len(train_data_loader))
        loss_d_log/=(len(train_data_loader))
        l1_log/=(len(train_data_loader))
        gan_log/=(len(train_data_loader))
        unparied_log/=(len(train_data_loader))
        ave_psnr=val_log(net=net_g, val_data_loader=val_data_loader, device=device, data_vis=None)
        print('epoch=', epoch, round(ave_psnr, 2),round(ave_loss.__float__(), 6),'loss_D:',round(loss_d_log.__float__(), 6), 'epoch_time=', round(time.time()-t1, 2))
        if ave_loss<best_l1:
            best_l1=ave_loss
            torch.save({"epoch": epoch, "netG_state_dict": net_g.state_dict(),
                        "optimizer_state_dict": optimizer_g.state_dict(),
                        "lr_learning_rate": scheduler_g.state_dict()}, opt.save_model + 'netG_best_train.pth')
            torch.save({"epoch": epoch, "netD_state_dict": net_d.state_dict(),
                        "optimizer_state_dict": optimizer_d.state_dict(),
                        "lr_learning_rate": scheduler_d.state_dict()}, opt.save_model + 'netD_best_train.pth')
        torch.save({"epoch": epoch, "netG_state_dict": net_g.state_dict(),
                    "optimizer_state_dict": optimizer_g.state_dict(),
                    "lr_learning_rate": scheduler_g.state_dict()}, opt.save_model + 'netG_latest.pth')
        torch.save({"epoch": epoch, "netD_state_dict": net_d.state_dict(),
                    "optimizer_state_dict": optimizer_d.state_dict(),
                    "lr_learning_rate": scheduler_d.state_dict()}, opt.save_model + 'netD_latest.pth')
    print('best_psnr:', best_psnr)