import torch.nn as nn
from . import block as B
import torch
import numpy as np
from skimage import morphology
from cvc_model.my_cva import CVA


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

class CVCnet(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(CVCnet, self).__init__()

        self.fea_conv = nn.Sequential(B.conv_layer(in_nc, nf, kernel_size=3),ResB(nf),ResB(nf))
        # self.fea_conv=B.conv_layer(in_nc, nf, kernel_size=3)

        self.pam = PAM(64)
        # IMDBs
        # IMDBs
        self.SPM1 = B.SPM(in_channels=nf)
        self.SPM2 = B.SPM(in_channels=nf)
        self.SPM3 = B.SPM(in_channels=nf)
        self.SPM4 = B.SPM(in_channels=nf)
        self.SPM5 = B.SPM(in_channels=nf)
        self.SPM6 = B.SPM(in_channels=nf)

        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input,right):
        out_fea = self.fea_conv(input)
        right=self.fea_conv(right)

        # out_fea=self.conv_merge(torch.cat((out_fea,right), dim=1))
        out_fea=self.pam(out_fea,right)

        out_B1 = self.SPM1(out_fea)
        out_B2 = self.SPM2(out_B1)
        out_B3 = self.SPM3(out_B2)
        out_B4 = self.SPM4(out_B3)
        out_B5 = self.SPM5(out_B4)
        out_B6 = self.SPM6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))

        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)

        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
        self.cva_zxy = CVA(64, 64, 64, 64, 64, dropout=0.05, sizes=([1]),
                          norm_type='batchnorm')  # sync_batchnorm

    def __call__(self, x_left, x_right):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)#torch.Size([32, 64, 30, 90])

        cva=self.cva_zxy(buffer_right, buffer_left)

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W

        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))               # (B*H) * W * W  torch.Size([960, 90, 90])

        M_right_to_left = self.softmax(score)#右图到左图的视差注意力图

        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                # (B*H) * W * W torch.Size([960, 90, 90])

        M_left_to_right = self.softmax(score)#左图到右图的视差注意力图

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)

        ### fusion
        buffer = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out = self.fusion(torch.cat((buffer, x_left, V_left_to_right), 1))

        return out+cva

def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)



