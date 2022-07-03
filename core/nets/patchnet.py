import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# implemented by https://github.com/naver/r2d2/blob/master/nets/patchnet.py
class BaseNet (nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            # for sure in [0,1], much less plateaus than softmax
            return x / (1 + x)
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def sum_by_render_mask(self, x, render_mask):
        return torch.sum(x * render_mask, 0, keepdim=True)

    def normalize(self, x, ureliability, urepeatability, un):
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=ureliability if un == 1 else self.softmax(ureliability))

    def normalize2(self, x, ureliability, urepeatability, render_mask=None):
        if self.inter_obj_dim > 0:
            d_d, d_o = self.split_d_norm(x, self.inter_obj_dim)
            d_d = F.normalize(d_d, p=2, dim=1)
            d_o = F.normalize(d_o, p=2, dim=1)
            des_total = torch.cat((d_d, d_o), 1)

            d_o_logit = self.seg(d_o) if self.use_seg else None
        else:
            des_total = F.normalize(x, p=2, dim=1)
            d_o_logit = None

        if render_mask is not None:
            des_total = self.sum_by_render_mask(des_total, render_mask)
            d_o_logit = self.sum_by_render_mask(
                d_o_logit, render_mask) if self.use_seg else torch.tensor([0])
            urepeatability = self.sum_by_render_mask(
                urepeatability, render_mask)
            ureliability = self.sum_by_render_mask(ureliability, render_mask)

        return dict(descriptors=des_total,
                    seg_logits=d_o_logit,
                    repeatability=self.softmax(urepeatability),
                    reliability=ureliability)

    def forward_one(self, x, render_mask=None):
        raise NotImplementedError()

    def forward(self, input_feas_list, render_mask=None, **kw):
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


class PatchNet (BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd,
                        kernel_size=k, **conv_params))
        if bn and self.bn:
            self.ops.append(self._make_bn(outd))
        if relu:
            self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)


class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """

    def __init__(self, out_dim=128, inter_obj_dim=0, mchan=4, relu22=False,
                 use_depth=False, uncertainty=False, num_class=22, use_seg=False, **kw):
        # repeat_target='repeatability', use_relia_score=False, **kw):
        PatchNet.__init__(self)
        if use_depth:
            self.curchan = 6
        self._add_conv(8*mchan)
        self._add_conv(8*mchan)
        self._add_conv(16*mchan, stride=2)
        self._add_conv(16*mchan)
        self._add_conv(32*mchan, stride=2)
        self._add_conv(32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)  # ：kernel_size,
        self._add_conv(32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(out_dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = out_dim
        self.inter_obj_dim = inter_obj_dim
        self.un = uncertainty
        self.num_class = num_class
        self.use_seg = use_seg


class r2d2 (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """

    def __init__(self, **kw):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x, render_mask=None):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps

        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)

        return self.normalize(x, ureliability, urepeatability, 0)


class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """

    def __init__(self, **kw):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        #self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        #self.clf2 = nn.Conv2d(self.out_dim, 1, kernel_size=1)
        if self.un == 1:
            self.clf = nn.Conv2d(self.out_dim, 1, kernel_size=1)
        # else:
        #     self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)
        if self.use_seg:
            self.seg = nn.Conv2d(self.inter_obj_dim,
                                 self.num_class, kernel_size=1)

    def split_d_norm(self, des, inter_obj_dim):
        #B, C, H, W = des.shape
        d1 = des[:, :-inter_obj_dim, :, :]
        d2 = des[:, -inter_obj_dim:, :, :]
        d1 = F.normalize(d1, p=2, dim=1)
        d2 = F.normalize(d2, p=2, dim=1)
        return d1, d2

    def forward_one(self, x, render_mask=None):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)

        # compute the confidence maps
        ureliability = self.clf(x**2) if self.un == 1 else None
        urepeatability = self.sal(x**2)

        return self.normalize2(x, ureliability, urepeatability, render_mask)
