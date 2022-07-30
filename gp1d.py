from time import process_time
import argparse
import warnings
import numpy as np
import torch
import gpytorch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adam
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

num_epochs = 30
learning_rate = 1e-1
N = 3
TEST_SIZE = 10
TRAIN_SIZE = 30

# Kilian's wonderful notes on Gaussian Processes: https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote15.html
# inspiration from GP tutorial: https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb

def mse_loss(output, target):
    print('mse', target.size(), output.size())
    return torch.pow(torch.abs(target-output), 2).sum()

def fgsm(model, likelihood, x, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X with L_infinity norm"""
    x.requires_grad = True
    output = model(x)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    loss = -mll(output, y)
    model.zero_grad()
    loss.backward()
    return epsilon * x.grad.data.sign()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def fit(x_train, y_train, x_test, y_test, model, likelihood, opt, train_size, epsilon):
    model.train()
    likelihood.train()
    best_loss = float('inf')
    count = 0
    for epoch in range(num_epochs):
        try:
            opt.zero_grad()
            delta = fgsm(model, likelihood, x_train, y_train, epsilon)    
            y_pred = model(x_train + delta)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            loss = -mll(y_pred, y_train)
            loss.backward()
            opt.step()
        except:
            print('train_inputs', model.train_inputs, 'model', model)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            count = 0
        else:
            count += 1
        if count == 30: # early stopping
            break

    # calc test loss
    model = ExactGPModel(x_train, y_train, likelihood)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(x_test))
        test_loss = mse_loss(y_pred.loc, y_test)
    return test_loss / TEST_SIZE

def main():
    global BEST_MODEL_PATH #, num_dim
    # num_dim = args.ndim
    num_dim = 1
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0]
    BEST_MODEL_PATH = f'gp_{num_dim}d_gaussian.pt'
    global x_test
    x_test = torch.randn(TEST_SIZE)
    e_test = torch.randn(TEST_SIZE)
    y_test = x_test + e_test
    # x_test_mean_squared = torch.mean(x_test).item() ** 2
    print('x_test', x_test)
    print('y_test', y_test)

    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    tic = process_time()
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        print('eps', epsilon)
        for train_size in range(1, TRAIN_SIZE+1):
            print('train_size', train_size)
            temp = np.zeros(N)
            for j in range(N):
                x_train = torch.randn(train_size)
                e_train = torch.randn(train_size)
                y_train = x_train + e_train
                # initialize likelihood and model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(x_train, y_train, likelihood)
                # includes GaussianLikelihood parameters
                opt = Adam(model.parameters(), lr=learning_rate)
                test_loss = fit(x_train, y_train, x_test, y_test, model, likelihood, opt, train_size, epsilon)
                temp[j] = test_loss # / x_test_mean_squared
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean
    toc = process_time()
    print("elapsed time:", toc - tic)
    print("test_losses:", test_losses)

if __name__ == "__main__":
    main()
