import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F


# implemented by r2d2: https://github.com/naver/r2d2/blob/master/nets/sampler.py
class FullSampler(nn.Module):
    """ all pixels are selected
        - feats: keypoint descriptors  #me feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        - confs: reliability values
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mode = 'bilinear'
        self.padding = 'zeros'

    @staticmethod
    def _aflow_to_grid(aflow, H=None, W=None):
        if H == None or W == None:
            H, W = aflow.shape[2:]
        grid = aflow.permute(0, 2, 3, 1).clone()
        grid[:, :, :, 0] *= 2/(W-1)
        grid[:, :, :, 1] *= 2/(H-1)
        grid -= 1
        grid[torch.isnan(grid)] = 9e9  # invalids

        return grid

    def _warp(self, feats, confs, aflow):
        if isinstance(aflow, tuple):
            return aflow  # result was precomputed
        feat1, feat2 = feats
        conf1, conf2 = confs if confs else (None, None)

        B, two, H, W = aflow.shape
        D = feat1.shape[1]
        assert feat1.shape == feat2.shape == (B, D, H, W)  # D = 128, B = batch
        assert conf1.shape == conf2.shape == (B, 1, H, W) if confs else True

        # warp img2 to img1
        grid = self._aflow_to_grid(aflow)
        ones2 = feat2.new_ones(feat2[:, 0:1].shape)
        feat2to1 = F.grid_sample(
            feat2, grid, mode=self.mode, padding_mode=self.padding, align_corners=True)
        mask2to1 = F.grid_sample(
            ones2, grid, mode='nearest', padding_mode='zeros', align_corners=True)  # 边界内的点
        conf2to1 = F.grid_sample(conf2, grid, mode=self.mode, padding_mode=self.padding, align_corners=True) \
            if confs else None
        return feat2to1, mask2to1.byte(), conf2to1

    def _warp_positions(self, aflow):
        B, two, H, W = aflow.shape
        assert two == 2

        Y = torch.arange(H, device=aflow.device)
        X = torch.arange(W, device=aflow.device)
        XY = torch.stack(torch.meshgrid(Y, X)[::-1], dim=0)
        XY = XY[None].expand(B, 2, H, W).float()

        grid = self._aflow_to_grid(aflow)
        XY2 = F.grid_sample(XY, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=True)
        return XY, XY2
