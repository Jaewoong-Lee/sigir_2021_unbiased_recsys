"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import yaml
import warnings
import os

from time import time
import numpy as np
import pandas as pd
import tensorflow as tf

from data.preprocessor import preprocess_dataset
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m',  default='mf-du', type=str, required=False, \
                choices=['mf', 'rel-mf', 'mf-du'])
parser.add_argument('--preprocess_data', default=False, required=False, action='store_true')
parser.add_argument('--data', default='coat', type=str, required=False, choices=['coat', 'yahoo'])
parser.add_argument('--learning_rate', '-lr', default=0.01, type=float, required=False)
parser.add_argument('--regularization', '-reg', default=0.0000001, type=float, required=False)
parser.add_argument('--random_state', '-ran',  nargs='+', default=[10], required=False)
parser.add_argument('--hidden', '-hidden',  nargs='+', default=[128], required=False)
parser.add_argument('--alpha', '-alpha', default=0.5, type=float, required=False) # for positive propensity (propensity for clicked data)
parser.add_argument('--beta', '-beta', default=0.5, type=float, required=False) # for negative propensity (propensity for unclicked data)
parser.add_argument('--clip', '-clip', default=0.1, type=float, required=False)
parser.add_argument('--max_epoch', type=int, default=500, help='number of max epochs to train')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--threshold', type=int, default=4, help='binalized threshold')
if __name__ == "__main__":
    start_time = time()
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    print("------------------------------------------------------------------------------")
    print(args)
    print("------------------------------------------------------------------------------")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model_name = args.model_name

    # run simulations
    if args.preprocess_data:
        preprocess_dataset(data=args.data, threshold=args.threshold, alpha=args.alpha, beta=args.beta)

    trainer = Trainer(data=args.data, random_state=args.random_state, hidden=args.hidden, max_iters=args.max_epoch, lam=args.regularization, batch_size=args.batch_size, clip=args.clip,
                      eta=args.learning_rate, model_name=model_name)

    trainer.run()

    print('\n', '=' * 25, '\n')
    print(f'Finished Running {model_name}!')
    print(f'Total time: {time() - start_time}')
    print('\n', '=' * 25, '\n')
