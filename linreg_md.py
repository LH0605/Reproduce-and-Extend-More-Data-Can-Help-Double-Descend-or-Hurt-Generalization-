from time import process_time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt

mu = 0
sigma = 1
learning_rate = 1e-1
N = 300
TEST_SIZE = 100
TRAIN_SIZE = 50


def train_loss(model, x, y, epsilon):
    # return nn.MSELoss(reduction="sum")(output, target)
    return torch.pow(torch.abs(y-model(x.float())), 2)

def test_loss(weight):
    return torch.pow(torch.abs(weight-1.0), 2).sum()

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    x.requires_grad = True
    output = model(x)
    loss = train_loss(model, x, output, epsilon)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fit(num_epochs, train_loader, test_loader, model, opt, attack, train_size, epsilon):
    model.train()
    best_loss = float('inf')
    count = 0
    for epoch in range(num_epochs):
        sum_loss = 0
        for x, y in train_loader:
            opt.zero_grad()
            if attack == "opt":
                loss = torch.pow(torch.abs(y-model(x.float())) + epsilon * torch.norm(model.weight, 1), 2).sum()
            elif attack == "fgsm":
                delta = fgsm(model, x, y, epsilon)
                # perturbed training data
                x_pert = x + delta
                # predicted output
                y_pred = model(x_pert)
                loss = train_loss(model, x, y_pred, epsilon)
            elif attack == "pgd":
                pass
            loss.backward()
            opt.step()
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

    # calc test loss
    sum_test_loss = 0
    model = nn.Linear(num_dim, 1, bias=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
    model.eval()
    weight = model.weight
    for x, y in test_loader:
        loss = test_loss(weight)
        sum_test_loss += loss
    return sum_test_loss / TEST_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--ndim', default=1, type=int)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['opt', 'fgsm', 'pgd'])
    parser.add_argument('--l2', default=0, type=float)
    args = parser.parse_args()
    print('args: ', args)
    global BEST_MODEL_PATH, num_epochs, num_dim
    num_dim = args.ndim
    e_test = torch.randn(TEST_SIZE)
    if args.gaussian:
        epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0]
        BEST_MODEL_PATH = f'best_lreg_{num_dim}d_{args.attack}_{round(args.l2, 2)}_model_gaussian.pt'
        num_epochs = 300
        x_test = torch.randn(TEST_SIZE, num_dim)
    else:
        epsilons = [0, 1., 2., 3., 4., 5., 6., 8., 10., 12.]
        BEST_MODEL_PATH = f'best_lreg_{num_dim}d_{args.attack}_{round(args.l2, 2)}_model_poisson.pt'
        num_epochs = 100
        x_test = (torch.distributions.poisson.Poisson(5).sample((TEST_SIZE, num_dim)) + 1).float()
    y_test = torch.unsqueeze(x_test.sum(1) + e_test, dim=1)
    test_set = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE, shuffle=False)

    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    vars = np.zeros((len(epsilons), TRAIN_SIZE))
    tic = process_time()
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            temp = np.zeros(N)
            for j in range(N):
                model = nn.Linear(num_dim, 1, bias=False)
                if args.l2:
                    opt = Adam(model.parameters(), lr=learning_rate, weight_decay=args.l2)
                else:
                    opt = Adam(model.parameters(), lr=learning_rate)
                e_train = torch.randn(train_size)
                if args.gaussian:
                    x_train = torch.randn(train_size, num_dim)
                else:
                    x_train = (torch.distributions.poisson.Poisson(5).sample((train_size, num_dim)) + 1).float()

                y_train = torch.unsqueeze(x_train.sum(1) + e_train, dim=1)
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
                
                test_loss = fit(num_epochs, train_loader, test_loader, model, opt, args.attack, train_size, epsilon)
                temp[j] = test_loss.item()
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean
            var = np.var(temp, dtype=np.float64)
            vars[i, train_size-1] = var
    toc = process_time()
    print("elapsed time:", toc-tic)
    print("test_losses:", test_losses)
    print("variance:", vars)

    # step = 3
    # train_sizes = np.arange(1, TRAIN_SIZE+1)
    # title_model = "Gaussian" if args.gaussian else "Poisson"
    # plt.title(f"Linear Regression 2D {title_model} (weak)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1:1+step])):
    #     epsilon = epsilons[1+i]
    #     plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"lrg_{num_dim}d_{title_model.lower()}_weak.png")
    # plt.clf()
    
    # plt.title(f"Linear Regression 2D {title_model} (medium)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1+step:1+(2*step)])):
    #     epsilon = epsilons[1+step+i]
    #     plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"lrg_{num_dim}d_{title_model.lower()}_medium.png")
    # plt.clf()
    
    # plt.title(f"Linear Regression 2D {title_model} (strong)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1+(2*step):])):
    #     epsilon = epsilons[1+(2*step)+i]
    #     plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"lrg_{num_dim}d_{title_model.lower()}_strong.png")

if __name__ == "__main__":
    main()
