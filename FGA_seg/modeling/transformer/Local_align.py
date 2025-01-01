import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

class Local_align(nn.Module):
    def __init__(self, word_dim=768, vision_dim=256, kernel_size=3):
        super().__init__()
        self.vision_dim = vision_dim
        self.kernel_size = kernel_size

        self.out_dim = 1 * self.vision_dim  * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, self.out_dim)
    
    def kernel_normalizer(self, mask0, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask0 = F.pixel_shuffle(mask0, self.scale_factor)
        n, mask_c = mask0.size()
        mask_channel = int(mask_c / float(kernel**2))
        mask = mask0.view(n, mask_channel, -1)
        mask = F.softmax(mask, dim=-1, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel)
        mask = mask.view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        return mask

    def forward(self, vision, text):
        '''
            x: b, C, H, W
            text: Nc, C
        '''
        # image
        img_feats = vision
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        B, C, H, W = img_feats.size() 
        # text
        text_feats = F.normalize(text, dim=-1) # B T P C
        text_feats = self.txt(text_feats)
        weight, bias = text_feats[:, :, :, :-1], text_feats[:, :, :, -1]
        # norm
        weight = [self.kernel_normalizer(weight[batch].squeeze(1), self.kernel_size).unsqueeze(0) for batch in range(B)]
        weight = torch.cat(weight, dim=0)

        print(f"weight_weight.shape_{weight.shape}")

        # local corr 
        align_out = []
        for batch in range(B):
            img_featsbb = img_feats[batch]
            weightbb = weight[batch]
            biasbb = bias[batch].squeeze(-1)
            out = F.conv2d(img_featsbb, weightbb, padding=self.kernel_size // 2, groups=1, bias=biasbb)
            align_out.append(out.unsqueeze(0))
            
        # transpose
        local_corr = torch.cat(align_out, dim=0)
        return local_corr     # b, Nc, H, W