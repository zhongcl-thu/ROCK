from .U_net import *
from .patchnet import *
from .superpoint import *


def model_entry(config, public_params):
    return globals()[config["type"]](**config["kwargs"], **public_params)
