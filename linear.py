import argparse
from time import process_time
from math import sqrt
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD

mu = 1
sigma = 2
learning_rate = 1e-1
num_epochs = 300
W = 1.
N = 300
TEST_SIZE = 100
TRAIN_SIZE = 40

class WeightClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = w.clamp(-W, W)
clipper = WeightClipper()

def linear_loss(output, target):
    return -output.t() @ target

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_infinity norm"""
    x.requires_grad = True
    output = model(x)
    loss = linear_loss(output, y)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fgm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_2 norm"""
    x.requires_grad = True
    output = model(x)
    loss = linear_loss(output, y)
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    norm = grad.norm(dim=1, p=2)[:, None]
    return epsilon * sqrt(num_dim) * grad/norm

def pgd(model, x, y, epsilon):
    alpha = epsilon / 3.
    num_iter = 30
    delta = torch.zeros_like(x)
    adv_x = x.clone().detach()
    for _ in range(num_iter):
        adv_x.requires_grad = True
        output = model(adv_x)
        model.zero_grad()
        loss = linear_loss(output, y)
        loss.backward()
        grad = adv_x.grad.data
        adv_x = adv_x.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = x + delta
    return delta

def fit(num_epochs, train_loader, test_loader, model, opt, attack, train_size, epsilon):
    model.train()
    best_loss = float('inf')
    count = 0
    for epoch in range(num_epochs):
        sum_loss = 0
        for x, y in train_loader:
            opt.zero_grad()
            if attack == "opt":
                y_pred = model(x.view(x.shape[0], -1))[:,0] - y*epsilon*model.weight.norm(1)
            elif attack == "fgsm":
                delta = fgsm(model, x, y, epsilon)
                y_pred = model(x + delta)
            elif attack == "fgm":
                delta = fgm(model, x, y, epsilon)
                y_pred = model(x + delta)
            elif attack == "pgd":
                delta = pgd(model, x, y, epsilon)
                y_pred = model(x + delta)
            loss = linear_loss(y_pred, y)
            loss.backward()
            opt.step()
            model.apply(clipper)
            sum_loss += float(loss)
        epoch_train_loss = sum_loss / train_size
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            count = 0
        else:
            count += 1
        if count == 30: # early stopping
            break
    sum_test_loss = 0
    model = nn.Linear(num_dim, 1, bias=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    for x, y in test_loader:
        y_pred = model(x)
        loss = linear_loss(y_pred, y)
        sum_test_loss += loss
    temp = sum_test_loss / TEST_SIZE
    return temp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndim', type=int)
    parser.add_argument('--attack', default="fgsm", type=str, choices=['opt', 'fgsm', 'fgm', 'pgd'])
    parser.add_argument('--l2', default=0, type=float)
    args = parser.parse_args()
    print('args: ', args)
    global num_dim, BEST_MODEL_PATH
    num_dim = args.ndim
    BEST_MODEL_PATH = f'lin_{args.attack}_{num_dim}D_{int(args.l2*100)}_model.pt'
    print(BEST_MODEL_PATH)
    epsilons = [0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.3, 1.5, 1.7, 2.0]
    # epsilons = [0, 0.3, 0.5, 0.7, 1.1, 1.3, 1.5, 2.5, 2.7, 3.0]

    x_test, y_test = datasets.make_blobs(n_samples=TEST_SIZE, n_features=num_dim,
        centers=[np.full((num_dim),-mu) ,np.full((num_dim),mu)],cluster_std=sigma)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    y_test[y_test==0] = -1

    test_set = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE, shuffle=False)

    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    mins = np.zeros((len(epsilons), TRAIN_SIZE))
    maxs = np.zeros((len(epsilons), TRAIN_SIZE))
    tic = process_time()
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            temp = np.zeros(N)
            for j in range(N):
                model = nn.Linear(num_dim, 1, bias=False)
                opt = SGD(model.parameters(), lr=learning_rate, weight_decay=args.l2)
                x_train, y_train = datasets.make_blobs(n_samples=train_size, n_features=num_dim,
                    centers=[np.full((num_dim),-mu) ,np.full((num_dim),mu)],cluster_std=sigma)
                x_train = torch.FloatTensor(x_train)
                y_train = torch.FloatTensor(y_train)
                y_train[y_train==0] = -1
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, test_loader, model, opt, args.attack, train_size, epsilon)
                temp[j] = test_loss.item()
            mn = np.amin(temp)
            mx = np.amax(temp)
            mins[i, train_size-1] = mn
            maxs[i, train_size-1] = mx
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean
    toc = process_time()
    print("elapsed time:", toc - tic)
    print("mins:", mins)
    print("maxs:", maxs)
    print("test_losses:", test_losses)

if __name__ == "__main__":
    main()
