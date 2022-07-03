import sys
import argparse

sys.path.append("./")
from core.utils import Config
from core.test import Evaluator
import torch
import ipdb

parser = argparse.ArgumentParser(description="rock")
parser.add_argument("--test_model_root", default="", type=str)
parser.add_argument("--recover", action="store_true")
parser.add_argument("--config", default="", type=str)
parser.add_argument('--test_type', default='sim2real_6dpose', type=str)
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

def main():

    args = parser.parse_args()
    
    C = Config(args.config)
    args.device = 'cuda:0'
    torch.cuda.set_device(0)

    S = Evaluator(C)
    
    if args.test_type == 'cal_add':
        S.combine_all_obj_add()
        return

    S.initialize(args)

    if args.test_type == 'sim2real_6dpose':
        S.sim2real_pose_estimation()
    elif args.test_type == 'match_tracking' or \
            args.test_type == 'match_tracking_unseen':
        S.match_tracking()
    elif args.test_type == 'sim2real_match':
        S.sim2real_match()
    

if __name__ == "__main__":
    main()
