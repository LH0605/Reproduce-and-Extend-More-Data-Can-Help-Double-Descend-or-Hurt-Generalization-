import argparse
from time import process_time
import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adam
import matplotlib.pyplot as plt

learning_rate = 1e-1
num_epochs = 100
N = 300
TEST_SIZE = 100
TRAIN_SIZE = 30

def train_hinge_loss(x, y, weight, bias):
    loss = torch.mean(torch.clamp(1 - y * (x @ weight - bias), min=0))
    loss += c * ((weight.t() @ weight) / 2.0).item()
    return loss

def test_hinge_loss(x, y, weight, bias):
    return torch.mean(torch.clamp(1 - y * (x @ weight - bias), min=0))

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_infinity norm"""
    x.requires_grad = True
    output = model(x)
    loss = train_hinge_loss(x, y, model.weight.t(), model.bias)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fgm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_2 norm"""
    x.requires_grad = True
    output = model(x)
    loss = train_hinge_loss(x, y, model.weight.t(), model.bias)
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    norm_x = torch.norm(grad, 2)
    return epsilon * grad/norm_x

def pgd(model, x, y, epsilon):
    alpha = epsilon / 3.
    num_iter = 30
    delta = torch.zeros_like(x)
    adv_x = x.clone().detach()
    for _ in range(num_iter):
        adv_x.requires_grad = True
        output = model(adv_x)
        model.zero_grad()
        loss = train_hinge_loss(adv_x, y, model.weight.t(), model.bias)
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
            if attack == "fgsm":
                delta = fgsm(model, x, y, epsilon)
            elif attack == "fgm":
                delta = fgm(model, x, y, epsilon)
            elif attack == "pgd":
                delta = pgd(model, x, y, epsilon)
            x_pert = x + delta
            y_pred = model(x_pert)
            loss = train_hinge_loss(x_pert, y, model.weight.t(), model.bias)
            loss.backward()
            opt.step()
            sum_loss += float(loss)
        epoch_train_loss = sum_loss
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            count = 0
        else:
            count += 1
        if count == 30: # early stopping
            break

    # calc test loss
    sum_test_loss = 0
    model = nn.Linear(num_dim, 1)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    weight = model.weight.t()
    bias = model.bias
    for x, y in test_loader:
        loss = test_hinge_loss(x, y, weight, bias)
        sum_test_loss += loss
    return sum_test_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=float)
    parser.add_argument('--ndim', type=int)
    parser.add_argument('--attack', default="fgsm", type=str, choices=['fgsm', 'fgm', 'pgd'])
    args = parser.parse_args()
    print('args: ', args)
    global c, num_dim, BEST_MODEL_PATH
    c = args.c
    num_dim = args.ndim
    BEST_MODEL_PATH = f'svm_{args.attack}_{num_dim}D_{args.c}.pt'
    mu = torch.ones(num_dim)
    sigma = torch.eye(num_dim)
    epsilons = [0, 0.4, 0.6, 0.8, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    x_test = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((TEST_SIZE,)), torch.distributions.MultivariateNormal(mu, sigma).sample((TEST_SIZE,))]).float()	
    y_test = torch.unsqueeze(torch.cat([-torch.ones(TEST_SIZE), torch.ones(TEST_SIZE)]), dim=1).float()	
    test_set = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)
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
                model = nn.Linear(num_dim, 1)
                opt = Adam(model.parameters(), lr=learning_rate)
                x_train = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((train_size,)), torch.distributions.MultivariateNormal(mu, sigma).sample((train_size,))]).float()	
                y_train = torch.unsqueeze(torch.cat([-torch.ones(train_size), torch.ones(train_size)]), dim=1).float()
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, test_loader, model, opt, args.attack, train_size, epsilon)
                temp[j] = test_loss.item()
            mn = np.amin(temp)
            mx = np.amax(temp)
            mins[i, train_size-1] = mn
            maxs[i, train_size-1] = mx
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean.item()
    toc = process_time()
    print("elapsed time:", toc - tic)
    print("mins:", mins)
    print("maxs:", maxs)
    print("test_losses:", test_losses)

    step = 3
    train_sizes = np.arange(1, TRAIN_SIZE+1)
    plt.title(f"SVM {num_dim}D with {args.attack.upper()} c={args.c} weak")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1:1+step])):
        epsilon = epsilons[1+i]
        plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_{args.attack}_{num_dim}D_{args.c}_weak.png")
    plt.clf()

    plt.title(f"SVM {num_dim}D with {args.attack.upper()} c={args.c} medium")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+step:1+(2*step)])):
        epsilon = epsilons[1+step+i]
        plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_{args.attack}_{num_dim}D_{args.c}_medium.png")
    plt.clf()
    
    plt.title(f"SVM {num_dim}D with {args.attack.upper()} c={args.c} strong")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+(2*step):])):
        epsilon = epsilons[1+(2*step)+i]
        plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_{args.attack}_{num_dim}D_{args.c}_strong.png")

if __name__ == "__main__":
    main()
