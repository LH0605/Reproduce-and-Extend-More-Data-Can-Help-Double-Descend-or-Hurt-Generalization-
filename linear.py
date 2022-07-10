import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt

mu = 1
sigma = 2
learning_rate = 1e-1
num_epochs = 1000
num_features = 1
W = 2.
N = 500
TEST_SIZE = 100
TRAIN_SIZE = 30
BEST_MODEL_PATH = 'best_gaussian_mixture_linear_loss_model.pt'

class WeightClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = w.clamp(-W, W)
clipper = WeightClipper()

# x_test = torch.unsqueeze(torch.cat([torch.distributions.Normal(-mu, sigma).sample((TEST_SIZE,)), torch.distributions.Normal(mu, sigma).sample((TEST_SIZE,))]), dim=1).float()
# y_test = torch.unsqueeze(torch.cat([-torch.ones(TEST_SIZE), torch.ones(TEST_SIZE)]), dim=1).float()

x_test, y_test = datasets.make_blobs(n_samples=TEST_SIZE, n_features=num_features,
    centers=[np.full((num_features),-1) ,np.full((num_features),1)],cluster_std=sigma)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
y_test[y_test==0] = -1

test_set = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_set, batch_size=TEST_SIZE//10, shuffle=False)

def linear_loss(output, target):
    # return -output.t() @ target
    return torch.mean(torch.mul(-output, target))

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
    best_loss = float('inf')
    for epoch in range(num_epochs):
        sum_loss = 0
        for x, y in train_loader:
            opt.zero_grad()
            # delta = fgsm(model, x, y, loss_fn, 0.3)
            y_pred = model(x.view(x.shape[0], -1))[:,0] - y*epsilon*model.weight.norm(1)
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
    temp = sum_test_loss
    return temp

def main():
    loss_fn = linear_loss
    epsilons = [0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9]
    # epsilons = [0, 0.3, 0.5, 0.7, 1.1, 1.3, 1.5, 2.5, 2.7, 3.0]
    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
        # if True:
        #     train_size = 5
            
            temp = np.zeros(N)
            for j in range(N):
                model = nn.Linear(1, 1, bias=False)
                opt = Adam(model.parameters(), lr=learning_rate)
                # batch_size = min(5, train_size)
                # x_train = torch.unsqueeze(torch.cat([torch.distributions.Normal(-mu, sigma).sample((train_size,)), torch.distributions.Normal(mu, sigma).sample((train_size,))]), dim=1).float()
                # y_train = torch.unsqueeze(torch.cat([-torch.ones(train_size), torch.ones(train_size)]), dim=1).float()
                x_train, y_train = datasets.make_blobs(n_samples=train_size, n_features=num_features,
                            centers=[np.full((num_features),-1) ,np.full((num_features),1)],cluster_std=sigma)
                x_train = torch.FloatTensor(x_train)
                y_train = torch.FloatTensor(y_train)
                y_train[y_train==0] = -1
                train_set = Data.TensorDataset(x_train, y_train)
                train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
                test_loss = fit(num_epochs, train_loader, model, loss_fn, opt, train_size, epsilon)
                temp[j] = test_loss.item()
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean.item()
    print("test_losses:", test_losses)
    
    # step = 3
    # train_sizes = np.arange(1, TRAIN_SIZE+1)
    # plt.title("Gaussian Mixture with Linear Loss (weak)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1:1+step])):
    #     epsilon = epsilons[1+i]
    #     plt.plot(train_sizes, test_losses[1+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"linear_weak.png")
    # plt.clf()
    
    # plt.title("Gaussian Mixture with Linear Loss (medium)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1+step:1+(2*step)])):
    #     epsilon = epsilons[1+step+i]
    #     plt.plot(train_sizes, test_losses[1+step+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"linear_medium.png")
    # plt.clf()
    
    # plt.title("Gaussian Mixture with Linear Loss (strong)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1+(2*step):])):
    #     epsilon = epsilons[1+(2*step)+i]
    #     plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"linear_strong.png")

if __name__ == "__main__":
    main()
