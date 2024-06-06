import os
from torch import nn

import torch
from  networks import MutiTransformer
from data_operation import DatasetFromFolder_in_test_mode
from base_option import BaseOptions
from torch.utils.data import DataLoader

from util import one_image_from_GPU_tensor,PSNR
if __name__ == '__main__':
    opt = BaseOptions().init().parse_args()
    device=torch.device(opt.device)

    #create model
    net_g=MutiTransformer(resolution=opt.resolution)
    net_g=net_g.to(device)
    optimizer=torch.optim.Adam(net_g.parameters(),lr=opt.lr,betas=(opt.beta,0.999))
    test_dataset = DatasetFromFolder_in_test_mode(opt.testset,getname=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2)
    loss_l1 = nn.L1Loss().to(device)
    # Load parameters
    if os.path.isfile(opt.save_model+'DDformer.pth'):
        checkpoint = torch.load(opt.save_model+'DDformer.pth')
        start_epoch = checkpoint['epoch']
        print(start_epoch)
        net_g.load_state_dict(checkpoint['netG_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #test
    net_g.eval()
    opt.image_save_status=True
    if opt.image_save_status and not os.path.exists(opt.save_image):
        os.makedirs(opt.save_image)
    for iter,batch in enumerate(test_loader):
        with torch.no_grad():
            input,image_name=batch[0].to(device),batch[1][0]
            restord_image=net_g(input)
            if opt.image_save_status:
                restord_save=one_image_from_GPU_tensor(restord_image)
                restord_save.save(opt.save_image+image_name)

