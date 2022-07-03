import os
import logging
import yaml
import json

import torch
import torch.nn.functional as F

import core.nets as nets


class Config(object):
    def __init__(self, config_file):

        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config_path = config_file

        self.config = config
        self.config_file = config_file


def accuracy(output, target, topk=(1,), v_n=256):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = v_n

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous(
        ).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_logger(name, log_file, local_rank):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)6s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def aflow_to_grid(aflow, H=None, W=None):
    if H == None or W == None:
        H, W = aflow.shape[2:]
    grid = aflow.permute(0, 2, 3, 1).clone()
    grid[:, :, :, 0] *= 2/(W-1)
    grid[:, :, :, 1] *= 2/(H-1)
    grid -= 1
    grid[torch.isnan(grid)] = 9e9  # invalids

    return grid


def get_corr_grid(aflow, label1, label2):
    # batch
    _, two, _, _ = aflow.shape
    assert two == 2

    grid = aflow_to_grid(aflow)
    # border_mask = torch.where(grid.abs() <= 1, True, False)
    # border_mask = border_mask[:, :, :, 0] * border_mask[:, :, :, 1]

    label12 = F.grid_sample(label2.unsqueeze(1).float(), grid, mode='nearest',
                            padding_mode='zeros', align_corners=True).int().squeeze(1)
    valid_mask = torch.where(label1 == label12, True, False)
    valid_mask *= torch.where(label1 > 0, True, False)  # * border_mask

    return valid_mask, grid


def gen_pix_coords(batch_size, height, width):
    Y = torch.arange(height)
    X = torch.arange(width)
    YX = torch.stack(torch.meshgrid(Y, X), dim=0)
    YX = YX[None].expand(batch_size, 2, height, width).float()
    return YX


def load_model(load_weights_folder, models_to_load, net):

    if len(models_to_load) == 0:
        checkpoint = torch.load(load_weights_folder, lambda a, b: a)
        net.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    elif models_to_load == ['extractor']:
        pretrained_dict = torch.load(load_weights_folder)
        weights = pretrained_dict['extractor']
        net.load_state_dict(weights)
    else:
        assert os.path.isdir(load_weights_folder), \
            "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}_best.pth".format(n))
            model_dict = net[n].state_dict()
            pretrained_dict = torch.load(path)

            new_pretrained_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    new_pretrained_dict[k] = v
                elif k[7:] in model_dict:
                    new_pretrained_dict[k[7:]] = v
            model_dict.update(new_pretrained_dict)

            net[n].load_state_dict(model_dict)


def store_json(data, path):
    with open(path, 'w') as fw:
        json.dump(data, fw)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        return data
