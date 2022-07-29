import argparse
from time import process_time
from copy import deepcopy
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adam
import matplotlib.pyplot as plt

learning_rate = 1e-1
num_epochs = 100
N = 10
TEST_SIZE = 100
TRAIN_SIZE = 30

# inspiration from https://gist.github.com/mlaves/c98cd4e6bcb9dbd4d0c03b34bacb0f65

class KernelSVM(torch.nn.Module):
    def __init__(self, x, kernel='rbf',
                 gamma_init=1.0, train_gamma=True):
        super().__init__()
        self._train_data_x = x
        self._kernel = self.rbf
        self._gamma = torch.nn.Parameter(torch.FloatTensor([gamma_init]), requires_grad=train_gamma)
        self.model = torch.nn.Linear(x.size(0), 1)
        self.weight = self.model.weight.t()
        self.bias = self.model.bias

    def rbf(self, x):
        y = self._train_data_x.repeat(x.size(0), 1, 1)
        return torch.exp(-self._gamma*((x[:,None]-y)**2).sum(dim=2))

    def forward(self, x):
        y = self._kernel(x)
        y = self.model(y)
        return y

def train_hinge_loss(model, x, y):
    # print('train', y * model(x))
    loss = torch.mean(torch.clamp(1 - y * model(x), min=0))
    # print('model.weight', model.weight.size(), )
    loss += c * ((model.weight.t() @ model.weight) / 2.0).item()
    return loss

def test_hinge_loss(model, x, y):
    # print('test', y, model(x))
    # print(y * model(x), torch.mean(torch.clamp(1 - y * model(x), min=0)).item())
    return torch.mean(torch.clamp(1 - y * model(x), min=0)).item()

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_infinity norm"""
    x.requires_grad = True
    output = model(x)
    loss = train_hinge_loss(model, x, y)
    model.zero_grad()
    loss.backward()
    # print('sign', x.grad.data.sign())
    return epsilon * x.grad.data.sign()

def pgd_l2(model, x, y, epsilon):
    iter = 30
    delta = torch.zeros_like(x)
    adv_x = x.clone().detach()
    last_output = None
    for i in range(iter):
        iter += 1
        adv_x.requires_grad = True
        output = model(adv_x)
        print('out', output)
        if i > 0 and not torch.all(torch.eq(last_output.data.sign(), output.data.sign())):
            print('last', last_output)
            break
        last_output = output.clone().detach()
        model.zero_grad()
        loss = train_hinge_loss(model, adv_x, y)
        loss.backward()
        grad = adv_x.grad.data
        adv_x = adv_x.detach() + output.detach() * grad/grad.norm(dim=1, p=2)[:, None]
    print('return', epsilon * delta)
    return epsilon * delta

def fit(num_epochs, x_train, y_train, x_test, y_test, model, opt, attack, epsilon):
    model.train()
    for epoch in range(num_epochs):
        opt.zero_grad()
        if attack == "fgsm":
            delta = fgsm(model, x_train, y_train, epsilon)
        elif attack == "pgd_l2":
            delta = pgd_l2(model, x_train, y_train, epsilon)
        x_pert = x_train + delta
        y_pred = model(x_pert)
        loss = train_hinge_loss(model, x_pert, y_train)
        # print('loss', loss)
        loss.backward()
        opt.step()

    # calc test loss
    model.eval()
    test_loss = test_hinge_loss(model, x_test, y_test)
    return test_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default=0.1, type=float)
    parser.add_argument('--attack', default="fgsm", type=str, choices=['fgsm', 'pgd_l2'])
    args = parser.parse_args()
    print('args: ', args)
    global c, BEST_MODEL_PATH
    c = args.c
    num_dim = 2
    BEST_MODEL_PATH = f'kernel_svm_{num_dim}D_{args.c}.pt'
    mu = torch.ones(num_dim)
    sigma = torch.eye(num_dim)
    epsilons = [0.4, 0.6, 0.8, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 9.0, 12.0, 15.0]
    x_test = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((TEST_SIZE,)), torch.distributions.MultivariateNormal(mu, sigma).sample((TEST_SIZE,))]).float()	
    y_test = torch.unsqueeze(torch.cat([-torch.ones(TEST_SIZE), torch.ones(TEST_SIZE)]), dim=1).float()	
    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    # mins = np.zeros((len(epsilons), TRAIN_SIZE))
    # maxs = np.zeros((len(epsilons), TRAIN_SIZE))
    tic = process_time()
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            temp = np.zeros(N)
            for j in range(N):
                # model = nn.Linear(num_dim, 1)
                x_train = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((train_size,)), torch.distributions.MultivariateNormal(mu, sigma).sample((train_size,))]).float()	
                y_train = torch.unsqueeze(torch.cat([-torch.ones(train_size), torch.ones(train_size)]), dim=1).float()
                
                model = KernelSVM(x_train)
                opt = Adam(model.parameters(), lr=learning_rate)
                # train_set = Data.TensorDataset(x_train, y_train)
                # train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
                test_loss = fit(num_epochs, x_train, y_train, x_test, y_test, model, opt, args.attack, epsilon)
                temp[j] = test_loss
            mn = np.amin(temp)
            mx = np.amax(temp)
            # mins[i, train_size-1] = mn
            # maxs[i, train_size-1] = mx
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean
    toc = process_time()
    print("elapsed time:", toc - tic)
    # print("mins:", mins)
    # print("maxs:", maxs)
    print("test_losses:", test_losses)

if __name__ == "__main__":
    main()
