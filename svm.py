import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt

mu = torch.ones(2)
sigma = torch.eye(2)
learning_rate = 1e-1
# epsilon = 3.
num_epochs = 100
c = 0.1
TEST_SIZE = 10000
TRAIN_SIZE = 30
BEST_MODEL_PATH = 'best_svm_model.pt'

x_test = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((TEST_SIZE,)), torch.distributions.MultivariateNormal(mu, sigma).sample((TEST_SIZE,))]).float()
y_test = torch.unsqueeze(torch.cat([-torch.ones(TEST_SIZE), torch.ones(TEST_SIZE)]), dim=1).float()
test_set = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)

def train_hinge_loss(x, y, weight, bias):
    loss = torch.mean(torch.clamp(1 - y * (x @ weight - bias), min=0))
    loss += c * ((weight.t() @ weight) / 2.0).item()
    return loss

def test_hinge_loss(x, y, weight, bias):
    return torch.mean(torch.clamp(1 - y * (x @ weight - bias), min=0))

def fgsm(model, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    x.requires_grad = True
    output = model(x)

    loss = train_hinge_loss(x, y, model.weight.t(), model.bias)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fit(num_epochs, train_loader, model, loss_fn, opt, train_size, epsilon):
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
            bias = model.bias
            loss = train_hinge_loss(x_pert, y, weight, bias)
            loss.backward()
            opt.step()
            sum_loss += float(loss)
        epoch_train_loss = sum_loss / train_size
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    # calc test loss
    sum_test_loss = 0
    model = nn.Linear(2, 1)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    weight = model.weight.t()
    bias = model.bias
    for x, y in test_loader:
        loss = test_hinge_loss(x, y, weight, bias)
        sum_test_loss += loss
    return sum_test_loss / TEST_SIZE

def main():
    epsilons = [0, 0.4, 0.6, 0.8, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            N = 100
            temp = np.zeros(N)
            for j in range(N):
            
                model = nn.Linear(2, 1)
                opt = Adam(model.parameters(), lr=learning_rate)
                batch_size = min(5, train_size)
                x_train = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((train_size,)), torch.distributions.MultivariateNormal(mu, sigma).sample((train_size,))]).float()
                y_train = torch.unsqueeze(torch.cat([-torch.ones(train_size), torch.ones(train_size)]), dim=1).float()
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, model, test_hinge_loss, opt, train_size, epsilon)
                
                temp[j] = test_loss.item()
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean.item()

    print("test_losses:", test_losses)

    step = 3
    train_sizes = np.arange(1, TRAIN_SIZE+1)
    plt.title("SVM with Hinge Loss (weak)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1:1+step])):
        epsilon = epsilons[1+i]
        plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_weak.png")
    plt.clf()

    plt.title("SVM with Hinge Loss (medium)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+step:1+(2*step)])):
        epsilon = epsilons[1+step+i]
        plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_medium.png")
    plt.clf()
    
    plt.title("SVM with Hinge Loss (strong)")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    for i in range(len(epsilons[1+(2*step):])):
        epsilon = epsilons[1+(2*step)+i]
        plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    plt.legend(loc="best")
    plt.savefig(f"svm_strong.png")

if __name__ == "__main__":
    main()
