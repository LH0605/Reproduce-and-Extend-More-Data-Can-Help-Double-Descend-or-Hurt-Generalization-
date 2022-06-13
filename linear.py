import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

mu = 1
sigma = 2
learning_rate = 1e-1
# epsilon = 1.5
num_epochs = 4
W = 3
TEST_SIZE = 50
TRAIN_SIZE = 30
BEST_MODEL_PATH = 'best_gaussian_mixture_linear_loss_model.pt'

class WeightClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            # W = 2.
            module.weight.data = w.clamp(-W, W)
clipper = WeightClipper()

x_test = torch.unsqueeze(torch.cat([torch.distributions.Normal(-mu, sigma).sample((TEST_SIZE,)), torch.distributions.Normal(mu, sigma).sample((TEST_SIZE,))]), dim=1).float()
y_test = torch.unsqueeze(torch.cat([-torch.ones(TEST_SIZE), torch.ones(TEST_SIZE)]), dim=1).float()
test_set = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)

def linear_loss(output, target):
    return -output.t() @ target

def fgsm(model, x, y, loss_fn, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    x.requires_grad = True
    output = model(x)
    loss = loss_fn(output, y)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

def fit(num_epochs, train_loader, model, loss_fn, opt, train_size, epsilon):
    model.train()
    for epoch in range(num_epochs):
        sum_loss = 0
        best_loss = float('inf')
        
        for x, y in train_loader:
            opt.zero_grad()
            delta = fgsm(model, x, y, loss_fn, epsilon)
            # perturbed training data
            x_pert = x + delta
            # predicted output
            y_pred = model(x_pert)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            model.apply(clipper)
            sum_loss += float(loss)
        epoch_train_loss = sum_loss / train_size
        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    sum_test_loss = 0
    model = nn.Linear(1, 1, bias=False)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    for x, y in test_loader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        sum_test_loss += loss
    temp = sum_test_loss / TEST_SIZE
    return temp

def main():
    loss_fn = linear_loss
    test_losses = np.zeros(TRAIN_SIZE)

    epsilons = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    for epsilon in epsilons:
        print("epsilon:", epsilon)
    # epsilon = 0.7
    # if True:
        for train_size in range(1, TRAIN_SIZE+1):
            N = 50
            temp = np.zeros(N)
            for i in range(N):
                model = nn.Linear(1, 1, bias=False)
                opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                batch_size = min(5, train_size)
                x_train = torch.unsqueeze(torch.cat([torch.distributions.Normal(-mu, sigma).sample((train_size,)), torch.distributions.Normal(mu, sigma).sample((train_size,))]), dim=1).float()
                y_train = torch.unsqueeze(torch.cat([-torch.ones(train_size), torch.ones(train_size)]), dim=1).float()
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, model, loss_fn, opt, train_size, epsilon)
                temp[i] = test_loss.item()
            mean = np.mean(temp)
            test_losses[train_size-1] = test_loss.item()

        print("test_losses:", test_losses)
        train_sizes = np.arange(1, TRAIN_SIZE+1)
        plt.plot(train_sizes, test_losses, label=f"Ɛ = {epsilon}")
        plt.title("Gaussian Mixture with Linear Loss")
        plt.xlabel("Size of Training Dataset")
        plt.ylabel("Test Loss")
        # plt.legend(loc='best')
        plt.savefig(f"linear.png")
        # plt.savefig(f"linear_{epsilon}.png")
        plt.clf()


if __name__ == "__main__":
    main()
