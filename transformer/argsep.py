import argparse
import os
def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument('--data',default = 'ETTh1')
    parser.add_argument('--freq',default = 'h')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--batch_size',default = 128)
    parser.add_argument('--root_path',default = '')
    parser.add_argument('--data_path',default = '')
    parser.add_argument('--features',default = 'S')
    parser.add_argument('--target',default = 'OT')
    parser.add_argument('--inverse',default = False)
    parser.add_argument('--seq_len',default = 24*4*4)
    parser.add_argument('--label_len',default = 24*4)
    parser.add_argument('--pred_len',default = 24*4)

    return args