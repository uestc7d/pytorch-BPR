'''
Created on Apr 12, 2016

@author: hexiangnan
'''
import torch
import numpy as np
from numpy import random
from torch.utils.data import Dataset

def LoadRatingFile_HoldKOut(filename, splitter, K):
    """
    Each line of .rating file is: userId(starts from 0), itemId, ratingScore, time
    Each element of train is the [[item1, time1], [item2, time2] of the user, sorted by time
    Each element of test is the [user, item, time] interaction, sorted by time
    """
    train = []  
    test = []
    
    # load ratings into train.
    num_ratings = 0
    num_item = 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(splitter)
            if (len(arr) < 4):
                continue
            user, item, time = int(arr[0]), int(arr[1]), long(arr[3]) 
            if (len(train) <= user):
                train.append([])
            train[user].append([item, time])
            num_ratings += 1
            num_item = max(item, num_item)
            line = f.readline()
    num_user = len(train)
    num_item = num_item + 1
    
    # sort ratings of each user by time
    def getTime(item):
        return item[-1];
    for u in range (len(train)):
        train[u] = sorted(train[u], key = getTime)
    
    # split into train/test
    for u in range (len(train)):
        for k in range(K):
            if (len(train[u]) == 0):
                break
            test.append([u, train[u][-1][0], train[u][-1][1]])
            del train[u][-1]    # delete the last element from train
            
    # sort the test ratings by time
    test = sorted(test, key = getTime)
    
    return train, test, num_user, num_item, num_ratings



# def pinterest(dir, splitter, K):
#     """
#     Each line of .rating file is: userId(starts from 0), itemId, ratingScore, time
#     Each element of train is the [[item1, time1], [item2, time2] of the user, sorted by time
#     Each element of test is the [user, item, time] interaction, sorted by time
#     """
#     train = []  
    
#     # load ratings into train.
#     num_ratings = 0
#     num_item = 0
#     with open(dir+'pos.txt', "r") as f:
#         line = f.readline()
#         while line != None and line != "":
#             arr = line.split(splitter)
#             if (len(arr) < 2):
#                 continue
#             user, item = int(arr[0]), int(arr[1])
#             if (len(train) <= user):
#                 train.append([])
#             train[user].append([item])
#             num_ratings += 1
#             num_item = max(item, num_item)
#             line = f.readline()
#     num_user = len(train)
#     num_item = num_item + 1

#     test = []
#     neg = dict()
#     # load ratings into test.
#     user = 0
#     with open(dir+'neg.txt', 'r') as f_neg:
#         line = f_neg.readline()
#         while line != None and line != '':
#             arr = line.split(splitter)
#             pos = int(arr[0])
#             test.append([user, pos])
#             neg[user] = []
#             for neg_i in xrange(len(arr)):
#                 if arr[neg_i] != '\n':
#                     neg[user].append(int(arr[neg_i]))

#             user += 1
#             line = f_neg.readline()
    
#     return train, test, num_user, num_item, num_ratings, neg\


class Pinterest(Dataset):
    def __init__(self, dir, splitter, K):
        """
        Each line of .rating file is: userId(starts from 0), itemId, ratingScore, time
        Each element of train is the [[item1, time1], [item2, time2] of the user, sorted by time
        Each element of test is the [user, item, time] interaction, sorted by time
        """

        self.train = []
        
        # load ratings into train.
        self.num_ratings = 0
        self.num_item = 0
        with open(dir+'pos.txt', "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(splitter)
                if (len(arr) < 2):
                    continue
                user, item = int(arr[0]), int(arr[1])
                if (len(self.train) <= user):
                    self.train.append([])
                self.train[user].append([item])
                self.num_ratings += 1
                self.num_item = max(item, self.num_item)
                line = f.readline()
        self.num_user = len(self.train)
        self.num_item = self.num_item + 1

        self.test = []
        self.neg = dict()
        # load ratings into test.
        user = 0
        with open(dir+'neg.txt', 'r') as f_neg:
            line = f_neg.readline()
            while line != None and line != '':
                arr = line.split(splitter)
                pos = int(arr[0])
                self.test.append([user, pos])
                self.neg[user] = []
                for neg_i in xrange(len(arr)):
                    if arr[neg_i] != '\n':
                        self.neg[user].append(int(arr[neg_i]))

                user += 1
                line = f_neg.readline()
        print("#users: %d, #items: %d, #ratings: %d" %(self.num_user, self.num_item, self.num_ratings))


    def __len__(self):
        return self.num_user


    def __getitem__(self, idx):
        u = idx
        i = self.train[u][np.random.randint(0, len(self.train[u]))]
        j = np.random.randint(0, self.num_item)
        while j in self.train[u]:
            j = np.random.randint(0, self.num_item) 
        
        return (u, i, j)
    


    
        