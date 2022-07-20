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

    # cross validation
    parser.add_argument('--cross_number', default=5, type=int)

    # data
    parser.add_argument('--test_dir', default="../sample_data/")
    parser.add_argument('--data_dir', default="../sample_data/")
    parser.add_argument('--p_data_dir', default="../sample_data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--save_data', default=False, type=str2bool)
    parser.add_argument('--load_saved_data', default=False, type=str2bool)
    parser.add_argument('--load_saved_test_data', default=False, type=str2bool)

    # perspectives
    parser.add_argument('--control_flow_p', default=True, type=str2bool)
    parser.add_argument('--time_p', default=True, type=str2bool)
    parser.add_argument('--resource_p', default=False, type=str2bool)
    parser.add_argument('--data_p', default=False, type=str2bool)
    parser.add_argument('--transition', default=False, type=str2bool)

    args = parser.parse_args()

    return args
