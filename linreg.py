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
num_epochs = 10
TEST_SIZE = 1000 # TODO: change to 5k!
TRAIN_SIZE = 30
low, high = 0., 2.


def train_loss(output, target):
    return nn.MSELoss(reduction="sum")(output, target)

def test_loss(weight, w_star):
    return nn.MSELoss()(weight, w_star)

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    x.requires_grad = True
    output = model(x)
    loss = train_loss(output, y)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fit(num_epochs, train_loader, test_loader, model, opt, train_size, epsilon, w_star):
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        sum_loss = 0
        for x, y in train_loader:
            opt.zero_grad()
            delta = fgsm(model, x, y, epsilon)
            # perturbed training data
            x_pert = x + delta
            # predicted output
            y_pred = model(x_pert)
        
            weight = model.weight
            loss = train_loss(y_pred, y)
            loss.backward()
            opt.step()
            sum_loss += float(loss)
        epoch_train_loss = sum_loss / train_size
        print('BEST_MODEL_PATH: ', BEST_MODEL_PATH)
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print('best w: ', weight)
        print("Epoch:", epoch)
        print("Training loss:", epoch_train_loss)

    # calc test loss
    sum_test_loss = 0
    model = nn.Linear(1, 1, bias=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    weight = model.weight
    for x, y in test_loader:
        loss = test_loss(weight, w_star)
        sum_test_loss += loss
    return sum_test_loss / TEST_SIZE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    args = parser.parse_args()
    print('args: ', args)
    global BEST_MODEL_PATH
    if args.gaussian:
        epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
        BEST_MODEL_PATH = 'best_lrg_model_gaussian.pt'
    else:
        epsilons = [0, 1., 2., 3., 4., 5., 6., 8., 10., 12.]
        BEST_MODEL_PATH = 'best_lrg_model_poisson.pt'
    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            N = 100
            temp = np.zeros(N)
            for j in range(N):
                model = nn.Linear(1, 1, bias=False)
                opt = Adam(model.parameters(), lr=learning_rate)
                batch_size = min(5, train_size)
                
                w_star = low + torch.rand(1) * (high - low)
                print('w_star: ', w_star)
                
                e_train = torch.randn(TRAIN_SIZE, 1)
                if args.gaussian:
                    x_train = torch.randn(TRAIN_SIZE, 1)
                else:
                    x_train = torch.unsqueeze(torch.distributions.poisson.Poisson(5).sample((TRAIN_SIZE,)) + 1, dim=1).float()
                y_train = w_star * x_train + e_train
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
                
                e_test = torch.randn(TEST_SIZE, 1)
                x_test = torch.randn(TEST_SIZE, 1)
                y_test = w_star * x_test + e_test
                test_set = Data.TensorDataset(x_test, y_test)
                test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)

                test_loss = fit(num_epochs, train_loader, test_loader, model, opt, train_size, epsilon, w_star)
                temp[j] = test_loss.item()
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean.item()
    print("test_losses:", test_losses)

    step = 3
    train_sizes = np.arange(1, TRAIN_SIZE+1)
    title_model = "Gaussian" if args.gaussian else "Poisson"
    plt.title(f"Linear Regression {title_model} (weak)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1:1+step])):
        epsilon = epsilons[1+i]
        plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"lrg_{title_model.lower()}_weak.png")
    plt.clf()
    
    plt.title(f"Linear Regression {title_model} (medium)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+step:1+(2*step)])):
        epsilon = epsilons[1+step+i]
        plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"lrg_{title_model.lower()}_medium.png")
    plt.clf()
    
    plt.title(f"Linear Regression {title_model} (strong)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+(2*step):])):
        epsilon = epsilons[1+(2*step)+i]
        plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"lrg_{title_model.lower()}_strong.png")

if __name__ == "__main__":
    main()
