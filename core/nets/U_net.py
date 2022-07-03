from __future__ import absolute_import, division, print_function

import ipdb
import torch.nn as nn
from core.tools import common
from .decoder import Decoder
from .resnet_encoder import ResnetEncoder, ResnetEncoder_fusion


class Unet(nn.Module):
    def __init__(self, layers=18, pretrained=True, blocks=3,
                 out_dim=96, inter_obj_dim=32, num_class=22,
                 uncertainty=True, segmentation=True,
                 use_depth=False, merge_depth='input',
                 use_rep=False, **kw):
        super(Unet, self).__init__()

        self.net = {}
        self.parameters_to_train = []

        if use_depth and merge_depth == 'middle':
            encoder_rgb = ResnetEncoder(
                layers, pretrained=pretrained, num_input_images=1, scales=blocks)
            self.parameters_to_train += list(encoder_rgb.parameters())
            # encoder_rgb.to('cuda')
            print(" ( Model size (encoder_rgb): {:.0f}K parameters )".format(
                common.model_size(encoder_rgb)/1000))

            encoder_depth = ResnetEncoder(
                layers, pretrained=pretrained, num_input_images=1, scales=blocks)
            self.parameters_to_train += list(encoder_depth.parameters())
            # encoder_depth.to('cuda')
            print(" ( Model size (encoder_depth): {:.0f}K parameters )".format(
                common.model_size(encoder_depth)/1000))

            self.net["encoder"] = ResnetEncoder_fusion(
                encoder_rgb, encoder_depth)
        else:
            num_input_images = 2 if use_depth and merge_depth == 'input' else 1
            self.net["encoder"] = ResnetEncoder(layers, pretrained=pretrained,
                                                num_input_images=num_input_images,
                                                scales=blocks)
            self.parameters_to_train += list(self.net["encoder"].parameters())
            # self.net["encoder"].to('cuda')
            print(" ( Model size (encoder_rgb): {:.0f}K parameters )".format(
                common.model_size(self.net["encoder"])/1000))

        self.net["decoder"] = Decoder(self.net["encoder"].num_ch_enc,
                                      scales=blocks,
                                      out_dim=out_dim,
                                      inter_obj_dim=inter_obj_dim,
                                      num_class=num_class,
                                      un=uncertainty,
                                      use_seg=segmentation,
                                      use_rep=use_rep)
        self.parameters_to_train += list(self.net["decoder"].parameters())
        # self.net["decoder"].to('cuda')
        print(" ( Model size (decoder): {:.0f}K parameters )".format(
            common.model_size(self.net["decoder"])/1000))

    def forward_unit(self, imgs, render_mask=None):
        features = self.net["encoder"](imgs)
        outputs = self.net["decoder"](features, render_mask)

        return outputs

    def forward(self, inputs):
        outputs = self.forward_unit(imgs=[inputs['img1'], inputs['img2']])

        if 'render_color_1' in inputs.keys():
            outputs_render = self.forward_unit(
                imgs=inputs['render_color_1'],
                render_mask=inputs['render_valid_1'])
            outputs_render['render_flag'] = True  # flag
        else:
            outputs_render = {}
            outputs['render_flag'] = False

        return outputs, outputs_render
