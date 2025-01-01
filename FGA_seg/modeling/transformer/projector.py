import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

class Projector(nn.Module):
    def __init__(self, word_dim=768, vision_dim=256, kernel_size=3):
        super().__init__()
        self.vision_dim = vision_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(vision_dim * 2, vision_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(vision_dim * 2, vision_dim, 3, padding=1),
            nn.Conv2d(vision_dim, vision_dim, 1))
        # textual projector
        self.out_dim = 1 * vision_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, self.out_dim)
    def forward(self, x, word):
        '''
            x: b, C, 24, 24
            word: Nc, C
        '''
        # word = word.squeeze(1)
        x = self.vis(x) # C减半 H, W乘以4  24*4
        B, C, H, W = x.size() 
        # 1, b*256, 104, 104
        # x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word).transpose(1,0)
        weight, bias = word[:, :, :-1], word[:, :, -1]
        weight = weight.reshape(weight.shape[0], weight.shape[1], C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        align_out_temp = []
        for batch in range(B):
            img_featsbb = x[batch].unsqueeze(0)
            weightbb = weight[batch]
            biasbb = bias[batch]
            out = [F.conv2d(img_featsbb,
                        weightbb[i,:,:,:].unsqueeze(0),
                        padding=self.kernel_size // 2,
                        groups=1, # Channel to Channel conv
                        bias=biasbb[i].unsqueeze(0))  # 添加到卷积结果中的偏置项
                    for i in range(weightbb.shape[0]) 
                    ]
            align_out_temp.append(torch.cat(out, dim=0)) 
        align_out = torch.cat(align_out_temp, dim=1)
        # downsample
        align_out = align_out.transpose(1, 0)
        # b, Nc, 96, 96
        return align_out