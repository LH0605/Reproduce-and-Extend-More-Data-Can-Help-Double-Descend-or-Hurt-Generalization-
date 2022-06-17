import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt

# NOTE: USE "Linear Regression.ipynb" RATHER THAN THIS FILE FOR NOW
mu = 0
sigma = 1
learning_rate = 1e-1
# epsilon = 3.
num_epochs = 100
c = 0.1
TEST_SIZE = 500 # TODO: change to 5k!
TRAIN_SIZE = 30
BEST_MODEL_PATH = 'best_linreg_model.pt'

low, high = 1, 5
w_star = low + torch.rand(1) * (high - low)
e = torch.randn(TEST_SIZE, 1)
x_test = torch.randn(TEST_SIZE, 1)
y_test = w_star * x_test + e
test_set = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)


def train_loss(output, target):
    return nn.MSELoss()(output, target)

def test_loss(weight):
    return nn.MSELoss()(weight, w_star)

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    x.requires_grad = True
    output = model(x)
    loss = train_loss(output, y)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fit(num_epochs, train_loader, model, opt, train_size, epsilon):
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
        
#             y_pred = model(x)
            weight = model.weight.t() # .squeeze()
            loss = train_loss(y_pred, y)
            loss.backward()
            opt.step()
            sum_loss += float(loss)
        epoch_train_loss = sum_loss / train_size
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
#         print("Epoch:", epoch)
#         print("Training loss:", sum_loss / train_size)

    # calc test loss
    sum_test_loss = 0
    model = nn.Linear(1, 1, bias=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    for x, y in test_loader:
        weight = model.weight.t()
        loss = test_loss(weight)
        sum_test_loss += loss
    #     print('Training loss: ', loss_fn(model(x_train), y_train))
#     print('Test loss: ', test_loss / test_size)
    return sum_test_loss / TEST_SIZE

def main():
    epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
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
                e = torch.randn(TRAIN_SIZE, 1)
                x_train = torch.randn(TRAIN_SIZE, 1)
                y_train = w_star * x_train + e
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, model, opt, train_size, epsilon)
                temp[j] = test_loss.item()
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean.item()

    print("test_losses:", test_losses)

    step = 3
    train_sizes = np.arange(1, TRAIN_SIZE+1)
    plt.title("Linear Regression Gaussian (weak)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1:1+step])):
        epsilon = epsilons[1+i]
        plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"linreg_gaussian_weak.png")
    plt.clf()
    
    plt.title("Linear Regression Gaussian (medium)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+step:1+(2*step)])):
        epsilon = epsilons[1+step+i]
        plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"linreg_gaussian_medium.png")
    plt.clf()
    
    plt.title("Linear Regression Gaussian (strong)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+(2*step):])):
        epsilon = epsilons[1+(2*step)+i]
        plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"linreg_gaussian_strong.png")

if __name__ == "__main__":
    main()
