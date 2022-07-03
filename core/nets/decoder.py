from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import ipdb
from collections import OrderedDict
import torch.nn.functional as F


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Decoder(nn.Module):
    '''
    out_dim: size of each descriptor
    inter_obj_dim: size of the inter-object descriptor
    un: if True, using contrastive learning with uncertainty
    use_seg: if True, using auxiliary segmentation task to 
                train inter-object descriptor
    use_rep: if True, using repeatability like r2d2 to filter the keypoint fisrt,
                else, only using reliability (uncertainty) to select keypoint.

    '''

    def __init__(self, num_ch_enc, scales=4, use_skips=True,
                 out_dim=96, inter_obj_dim=32,
                 num_class=22, un=True,
                 use_seg=True, use_rep=False):
        super(Decoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.out_dim = out_dim
        if scales == 4:
            self.num_ch_dec = np.array([out_dim, out_dim, out_dim, 256, 256])
        elif scales == 3:
            self.num_ch_dec = np.array([out_dim, out_dim, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(scales, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            if i == 0:
                self.convs[("upconv", i, 1)] = Conv3x3(num_ch_in, num_ch_out)
            else:
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.un = un
        self.use_seg = use_seg
        self.inter_obj_dim = inter_obj_dim
        self.num_class = num_class
        self.use_rep = use_rep

        if self.un:
            self.convs["clf"] = nn.Conv2d(self.out_dim, 1, kernel_size=1)

        if use_rep:
            self.convs["sal"] = nn.Conv2d(self.out_dim, 1, kernel_size=1)

        if self.use_seg:
            self.convs["seg"] = nn.Conv2d(
                self.inter_obj_dim, self.num_class, kernel_size=1)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def split_d_norm(self, des, inter_obj_dim):
        #B, C, H, W = des.shape
        d1 = des[:, :-inter_obj_dim, :, :]
        d2 = des[:, -inter_obj_dim:, :, :]
        d1 = F.normalize(d1, p=2, dim=1)
        d2 = F.normalize(d2, p=2, dim=1)
        return d1, d2

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            # for sure in [0,1], much less plateaus than softmax
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def sum_by_render_mask(self, x, render_mask):
        return torch.sum(x * render_mask, 0, keepdim=True)

    def normalize(self, x, ureliability, urepeatability, render_mask=None):
        if self.inter_obj_dim > 0:
            d_d, d_o = self.split_d_norm(x, self.inter_obj_dim)
            d_d = F.normalize(d_d, p=2, dim=1)
            d_o = F.normalize(d_o, p=2, dim=1)
            des_total = torch.cat((d_d, d_o), 1)

            d_o_logit = self.convs["seg"](d_o) if self.use_seg else None
        else:
            des_total = F.normalize(x, p=2, dim=1)
            d_o_logit = None

        if render_mask is not None:
            des_total = self.sum_by_render_mask(des_total, render_mask)
            d_o_logit = self.sum_by_render_mask(
                d_o_logit, render_mask) if self.use_seg else torch.tensor([0])

            if self.use_rep:
                urepeatability = self.sum_by_render_mask(
                    urepeatability, render_mask)
            if self.un:
                ureliability = self.sum_by_render_mask(
                    ureliability, render_mask)

        if self.use_rep:
            repeatability = self.softmax(urepeatability)
        else:
            repeatability = torch.ones_like(ureliability)

        return dict(descriptors=des_total,
                    seg_logits=d_o_logit,
                    repeatability=repeatability,
                    reliability=ureliability)

    def forward_one(self, input_features, render_mask=None):
        # decoder
        x = input_features[-1]
        for i in range(self.scales, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
                if x[0].shape[2] > x[1].shape[2]:
                    x[0] = x[0][:, :, 0:x[1].shape[2], :]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        # todo
        ureliability = self.convs["clf"](x**2) if self.un else None
        if self.use_rep:
            urepeatability = self.convs["sal"](x**2)
        else:
            urepeatability = None

        return self.normalize(x, ureliability, urepeatability, render_mask)

    def forward(self, input_feas_list, render_mask, **kw):
        if render_mask is None:
            res = [self.forward_one(input_feas, None)
                   for input_feas in input_feas_list]
            # merge all dictionaries into one
            res = {k: [r[k] for r in res if k in r]
                   for k in {k for r in res for k in r}}
        else:
            res = [self.forward_one(input_feas, render_mask[i])
                   for i, input_feas in enumerate(input_feas_list)]
            res = {k+'_render': torch.cat([r[k] for r in res if k in r], 0)
                   for k in {k for r in res for k in r}}

        return dict(res, **kw)
