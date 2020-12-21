import os
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load():
    parser = argparse.ArgumentParser()

    # training status
    parser.add_argument('--status', default="o")

    # prediction task
    parser.add_argument('--task', default="next_activity")

    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int) 

    # feature
    parser.add_argument('--contextual_info', default=False, type=str2bool)
    parser.add_argument('--inter_case_level', default='Level1')
    parser.add_argument('--transition', default=False, type=str2bool)

    # data
    parser.add_argument('--data_set', default="test.csv")
    parser.add_argument('--data_dir', default="../testsets/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")

    # perspectives
    parser.add_argument('--control_flow_p', default=True, type=str2bool)
    parser.add_argument('--time_p', default=True, type=str2bool)
    parser.add_argument('--resource_p', default=False, type=str2bool)
    parser.add_argument('--data_p', default=False, type=str2bool)

    args = parser.parse_args()

    return args
