import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sets import Set
from evaluate import evaluate_model

class MFbpr(nn.Module):
    '''
    BPR learning for MF model
    '''
    def __init__(self, dataset, factors, learning_rate, reg, init_mean, init_stdev):
        '''
        Constructor
        '''
        super(MFbpr, self).__init__()
        self.dataset = dataset
        self.train = dataset.train
        self.test = dataset.test
        self.num_user = dataset.num_user
        self.num_item = dataset.num_item
        self.neg = dataset.neg
        self.factors = factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.init_mean = init_mean
        self.init_stdev = init_stdev

        # user & item latent vectors
        # self.U = torch.normal(mean = self.init_mean * torch.ones(self.num_user, self.factors), std = self.init_stdev).requires_grad_()
        # self.V = torch.normal(mean = self.init_mean * torch.ones(self.num_item, self.factors), std = self.init_stdev).requires_grad_()
        self.U = nn.Embedding(self.num_user, self.factors).weight
        self.V = nn.Embedding(self.num_item, self.factors).weight

        # optim
        self.mf_optim = optim.Adam([self.U, self.V], lr=self.learning_rate)
        # self.mf_scheduler = optim.lr_scheduler.MultiStepLR(self.mf_optim, [30, 200], 0.1)
        # Each element is the set of items for a user, used for negative sampling
        self.items_of_user = []
        self.num_rating = 0     # number of ratings
        for u in xrange(len(self.train)):
            self.items_of_user.append(Set([]))
            for i in xrange(len(self.train[u])):
                item = self.train[u][i][0]
                self.items_of_user[u].add(item)
                self.num_rating += 1


    def forward(self, u, i, j):
        '''
        Args:
            u: user id. type=int or list.
            i: positive item id. type=int or list.
            j: negative item id. type=int or list.

        Returns:
            y_ui: predicted score between user and positive item.
            y_uj: predicted score between user and negative item.
            loss: BPR loss. It is the opposite of BPR-OPT.
        '''
        y_ui = torch.diag(torch.mm(self.U[u], self.V[i].t()))
        y_uj = torch.diag(torch.mm(self.U[u], self.V[j].t()))
        regularizer = self.reg * (torch.sum(self.U[u] ** 2) + torch.sum(self.V[i] ** 2) + torch.sum(self.V[j] ** 2))
        loss = regularizer - torch.sum(torch.log2(torch.sigmoid(y_ui - y_uj)))
        return y_ui, y_uj, loss

    def build_model(self, maxIter=100, num_thread=4, batch_size=32):
        # dataloader
        data_loader = DataLoader(self.dataset, batch_size=batch_size)

        # Training process
        print("Training MF-BPR with: learning_rate=%.4f, regularization=%.4f, factors=%d, #epoch=%d, batch_size=%d."
          %(self.learning_rate, self.reg, self.factors, maxIter, batch_size))
        t1 = time.time()
        iter_loss = 0
        for iteration in xrange(maxIter):
            # self.mf_scheduler.step()
            # Each training epoch
            for s, (users, items_pos, items_neg) in enumerate(data_loader):
                # sample a batch of users, positive samples and negative samples
                
                # zero grad
                self.mf_optim.zero_grad()
                # forward propagation
                y_ui, y_uj, loss = self.forward(users, items_pos, items_neg)
                iter_loss += loss
                # back propagation
                loss.backward()
                self.mf_optim.step()
            
            # check performance
            if iteration % 20 == 19:
            	t2 = time.time()
                topK = 20
                (hits, ndcgs) = evaluate_model(self, self.test, topK, num_thread)
                # save the hr and ndcg value.
                hr_mean = np.array(hits).mean()
                ndcg_mean = np.array(ndcgs).mean()
        
                print("Iter=%d [%.1f s] HitRatio@%d = %.4f, NDCG@%d = %.4f [%.1f s]"
                      %(iteration, (t2-t1) / 20, topK, hr_mean, topK, ndcg_mean, time.time()-t2))
                t1 = time.time()
                iter_loss = 0
            

    def predict(self, u, i):
        return np.inner(self.U[u].detach().numpy(), self.V[i].detach().numpy())

    def get_batch(self, batch_size):
        users, pos_items, neg_items = [], [], []
        for i in xrange(batch_size):
            # sample a user
            u = np.random.randint(0, self.num_user)
            # sample a positive item
            i = self.train[u][np.random.randint(0, len(self.train[u]))][0]
            # sample a negative item
            j = np.random.randint(0, self.num_item)
            while j in self.items_of_user[u]:
                j = np.random.randint(0, self.num_item)
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
        return (users, pos_items, neg_items)



