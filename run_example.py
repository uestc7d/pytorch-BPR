import multiprocessing as mp
from dataloader import LoadRatingFile_HoldKOut, pinterest
from MFbpr import MFbpr

if __name__ == '__main__':
    
    # Load data
    dataset = "./data/"
    splitter = " "
    hold_k_out = 1
    train, test, num_user, num_item, num_ratings, neg = pinterest(dataset, splitter, hold_k_out)
    print("#users: %d, #items: %d, #ratings: %d" %(num_user, num_item, num_ratings))
    
    # MFbpr parameters
    factors = 64
    learning_rate = 0.0003
    reg = 0.01
    init_mean = 0
    init_stdev = 0.01
    maxIter = 10000
    batch_size = 32
    num_thread = mp.cpu_count()
    print("#factors: %d, lr: %f, reg: %f, batch_size: %d" % (factors, learning_rate, reg, batch_size))
    
    # Run model
    bpr = MFbpr(train, test, num_user, num_item, neg,
                factors, learning_rate, reg, init_mean, init_stdev)
    bpr.build_model(maxIter, num_thread, batch_size=batch_size)

