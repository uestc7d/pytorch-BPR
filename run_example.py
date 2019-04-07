import multiprocessing as mp
from dataloader import *
from MFbpr import MFbpr
import argparse
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run BPR.")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Load data
    dataset = "./data/"
    splitter = " "
    hold_k_out = 1
    pinterest = Pinterest(dataset, splitter, hold_k_out)
    
    # MFbpr parameters
    factors = 64
    learning_rate = args.learning_rate
    reg = 0.01
    init_mean = 0
    init_stdev = 0.01
    maxIter = 10000
    batch_size = args.batch_size
    num_thread = mp.cpu_count()
    print("#factors: %d, lr: %f, reg: %f, batch_size: %d" % (factors, learning_rate, reg, batch_size))
    
    # Run model
    bpr = MFbpr(pinterest,
                factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(maxIter, num_thread, batch_size=batch_size)

    # save model
    np.save("out/u"+str(learning_rate)+".npy", bpr.U.detach().numpy())
    np.save("out/v"+str(learning_rate)+".npy", bpr.V.detach().numpy())

