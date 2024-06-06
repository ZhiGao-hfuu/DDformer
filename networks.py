import functools

import torch
from model import DLT,SLT
from torch import nn, autograd
import  math
def define_D( netd,input_nc=3,ndf=64,n_layers_D=4,  use_sigmoid=False):
    net = None
    norm_layer = nn.BatchNorm2d
    # print('norm: {}'.format(norm))
    if netd == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netd == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netd == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netd == 'mutiscale':
        net = MutiScaleDiscriminator(n_layers=n_layers_D, ndf=ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return net
class MutiScaleDiscriminator(nn.Module):
    def __init__(self,n_layers=4,ndf=64):
        super(MutiScaleDiscriminator, self).__init__()
        self.n_layers=n_layers
        ks=2
        input_dim=3
        for n in range(1,self.n_layers+1):
            out_dim=ndf*n
            self.add_module('down{}'.format(n),nn.Sequential(
                nn.Conv2d(input_dim, out_dim,kernel_size=4,stride=2,padding=1),nn.BatchNorm2d(out_dim),nn.LeakyReLU(0.2,True)
            ))
            if n<self.n_layers:
                self.add_module('scale{}'.format(n),nn.Conv2d(3,out_dim,kernel_size=ks,stride=ks))
                ks*=2
            input_dim=out_dim

        self.add_module('outc',nn.Sequential(
                nn.Conv2d(input_dim, 1,kernel_size=4,stride=1,padding=1),nn.Tanh()
            ))
    def forward(self,input):
        x=input
        for n in range(1,self.n_layers+1):
            down=getattr(self,'down{}'.format(n))
            x=down(x)
            if n<self.n_layers:
                scale = getattr(self, 'scale{}'.format(n))
                x_scale=scale(input)
                x=x+x_scale
        out=self.outc(x)
        return out

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class NLayerDiscriminator(nn.Module):
    def  __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class MutiTransformer(nn.Module):
    #Muti Transformer
    def __init__(self,resolution=512):
        super(MutiTransformer, self).__init__()
        self.down_channels = [3, 32, 32 * 2, 32 * 4, 32 * 8]
        #downsample
        self.down1 = DLTLayer(self.down_channels[0], self.down_channels[1], resolution=resolution)
        self.down2 = DLTLayer(self.down_channels[1], self.down_channels[2], resolution=resolution // 2)
        self.down3 = DLTLayer(self.down_channels[2], self.down_channels[3], resolution=resolution // 4)
        self.down4 = DLTLayer(self.down_channels[3], self.down_channels[4], resolution=resolution // 8)
        self.down5 = SLTLayer(self.down_channels[4], self.down_channels[4], resolution=resolution // 16)
        self.down6 = SLTLayer(self.down_channels[4], self.down_channels[4], resolution=resolution // 32)
        # upsample
        self.up6 = SLTLayer(self.down_channels[4], self.down_channels[4], resolution=resolution // 32,
                                        down_sample=False)
        self.up5 = SLTLayer(self.down_channels[4], self.down_channels[4], resolution=resolution // 16,
                                        down_sample=False)
        self.up4 = DLTLayer(self.down_channels[4], self.down_channels[3], resolution=resolution // 8,
                                    down_sample=False)
        self.up3 = DLTLayer(self.down_channels[3], self.down_channels[2], resolution=resolution // 4,
                                    down_sample=False)
        self.up2= DLTLayer(self.down_channels[2], self.down_channels[1], resolution=resolution // 2,
                                    down_sample=False)
        self.up1= DLTLayer(self.down_channels[1],32, resolution=resolution,
                                    down_sample=False)
        #FIU
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True),
        )
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1,
                      padding=0),
            nn.Tanh()
        )
    def forward(self, input):
        x = input
        #high-resolution
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        #low-resolution
        d5= self.down5(d4)
        d6= self.down6(d5)
        up6 = self.up6(d6)
        up5 = self.up5(up6 + d5)
        # high-resolution
        up4 = self.up4(up5 + d4)
        up3 = self.up3(up4 + d3)
        up2 = self.up2(up3 + d2)
        up1 = self.up1(up2 + d1)
        #FIU
        l_e = self.local_enhance(up1)
        out = self.outc(torch.cat((l_e, input), dim=1))
        return out

class SLTLayer(nn.Module):
    def __init__(self, input_dim, output_dim, resolution, down_sample=True):
        super(SLTLayer, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.parallel_dim = input_dim
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )
        else:
            self.parallel_dim = output_dim
            self.sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )

        self.middle_dim = max(3, self.parallel_dim // 8)
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels=self.parallel_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.parallel_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.parallel_dim),
            nn.LeakyReLU(0.2, True)
            )

        self.slt = SLT(dim=self.parallel_dim, input_size=resolution)

    def forward(self, x):
        if self.down_sample:
            # print('down')
            attn = self.slt(x)
            le = self.local_enhance(attn)
            out = self.sample(attn + le+x)
        else:
            # print('up')
            sample = self.sample(x)
            attn = self.slt(sample)
            le = self.local_enhance(sample)
            out = attn + le+sample
        return out

class DLTLayer(nn.Module):
    def __init__(self, input_dim, output_dim, resolution, down_sample=True):
        super(DLTLayer, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.parallel_dim = input_dim
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )
        else:
            self.parallel_dim = output_dim
            self.sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )

        self.middle_dim = max(3, self.parallel_dim // 8)
        self.local_enhance = nn.Sequential(
            nn.Conv2d(in_channels=self.parallel_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.parallel_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.parallel_dim),
            nn.LeakyReLU(0.2, True)
            )

        self.dlt = DLT(dim=self.parallel_dim, input_size=resolution)

    def forward(self, x):
        if self.down_sample:
            # print('down')
            attn = self.dlt(x)
            le=self.local_enhance(attn)
            out = self.sample(le+attn+x)
        else:
            # print('up')
            sample = self.sample(x)
            attn = self.dlt(sample)
            le=self.local_enhance(attn)
            out = le + attn+sample
        return out

class Generator_Simplicity(nn.Module):
    def __init__(self,resolution=256):
        super(Generator_Simplicity, self).__init__()
        self.down_channels = [3, 64, 64 * 2, 64 * 4, 64 * 8]
        #downsample
        self.down1 = FuattnSimplicity(self.down_channels[0], self.down_channels[1], resolution=resolution)
        self.down2 = FuattnSimplicity(self.down_channels[1], self.down_channels[2], resolution=resolution // 2)
        self.down3 = FuattnSimplicity(self.down_channels[2], self.down_channels[3], resolution=resolution // 4)
        self.down4 = FuattnSimplicity(self.down_channels[3], self.down_channels[4], resolution=resolution // 8)
        # upsample
        self.up1 = FuattnSimplicity(self.down_channels[4], self.down_channels[3], resolution=resolution // 8,
                                    down_sample=False)
        self.up2 = FuattnSimplicity(self.down_channels[3] * 2, self.down_channels[2], resolution=resolution // 4,
                                    down_sample=False)
        self.up3 = FuattnSimplicity(self.down_channels[2] * 2, self.down_channels[1], resolution=resolution // 2,
                                    down_sample=False)
        self.up4 = FuattnSimplicity(self.down_channels[1] * 2, self.down_channels[0], resolution=resolution,
                                    down_sample=False)
        self.outc = nn.Sequential(
            nn.Conv2d(in_channels=self.down_channels[0]* 2, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1,
                      padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        x = input
        # down
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # up
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat((u1, d3), dim=1))
        u3 = self.up3(torch.cat((u2, d2), dim=1))
        u4 = self.up4(torch.cat((u3, d1), dim=1))
        out = self.outc(torch.cat((u4, input), dim=1))
        return out
class FuattnSimplicity(nn.Module):
    def __init__(self, input_dim, output_dim, resolution, down_sample=True):
        super(FuattnSimplicity, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.parallel_dim = input_dim
            self.sample = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )
        else:
            self.parallel_dim = output_dim
            self.sample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_dim, out_channels=output_dim, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(0.2, True)
                )

        self.middle_dim = max(3, self.parallel_dim // 8)
        self.serial = nn.Sequential(
            nn.Conv2d(in_channels=self.parallel_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.middle_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.middle_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=self.parallel_dim,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.parallel_dim),
            nn.LeakyReLU(0.2, True)
            )

        self.fuattn = DLT(dim=self.parallel_dim, input_size=resolution)

    def forward(self, x):
        if self.down_sample:
            # print('down')
            attn = self.fuattn(x)
            serial=self.serial(attn)
            out = self.sample(serial+attn)
        else:
            # print('up')
            sample = self.sample(x)
            attn = self.fuattn(sample)
            serial=self.serial(attn)
            out = serial + attn
        return out


class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv1.weight, 1)
        nn.init.constant_(self.conv1.weight[0, 0, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 1, 1, 1], -8)
        nn.init.constant_(self.conv1.weight[0, 2, 1, 1], -8)
    def forward(self, x1):
            edge_map = self.conv1(x1)
            return edge_map