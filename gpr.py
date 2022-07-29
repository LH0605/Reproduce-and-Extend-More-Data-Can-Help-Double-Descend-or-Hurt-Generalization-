import argparse
from time import process_time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

num_epochs = 300
learning_rate = 1e-1
N = 300
TEST_SIZE = 100
TRAIN_SIZE = 30

class GP(nn.Module):
    def __init__(self, length_scale=1., noise_scale=1., amplitude_scale=1.):
        super().__init__()
        self.length_scale_ = nn.Parameter(torch.tensor(np.log(length_scale)))
        self.noise_scale_ = nn.Parameter(torch.tensor(np.log(noise_scale)))
        self.amplitude_scale_ = nn.Parameter(torch.tensor(np.log(amplitude_scale)))

    @property
    def length_scale(self):
        return torch.exp(self.length_scale_)

    @property
    def noise_scale(self):
        return torch.exp(self.noise_scale_)

    @property
    def amplitude_scale(self):
        return torch.exp(self.amplitude_scale_)
    
    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        y = self.y; L = self.L; alpha = self.alpha
        k = self.kernel_mat(self.X, x)
        v = torch.linalg.solve(L, k)
        mu = k.T.mm(alpha)
        var = self.amplitude_scale + self.noise_scale - torch.diag(v.T.mm(v))
        return mu, var

    def fit(self, X, y):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor."""
        D = X.shape[1]
        K = self.kernel_mat_self(X)
        L = torch.linalg.cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = -0.5 * y.T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi)
        self.X = X; self.y = y; self.L = L; self.alpha = alpha; self.K = K
        return marginal_likelihood

    def kernel_mat_self(self, X):
        sq = (X ** 2).sum(dim=1, keepdim=True)
        sqdist = sq + sq.T - 2 *X.mm(X.T)
        return self.amplitude_scale * torch.exp(- 0.5 * sqdist / self.length_scale) + self.noise_scale * torch.eye(len(X))

    def kernel_mat(self, X, Z):
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        Zsq = (Z ** 2).sum(dim=1, keepdim=True)
        sqdist = Xsq + Zsq.T - 2 * X.mm(Z.T)
        return self.amplitude_scale * torch.exp(- 0.5 * sqdist / self.length_scale)

    def fgsm(self, X, y, epsilon):
        """ Construct FGSM adversarial examples on the examples X with L_infinity norm"""
        X.requires_grad = True
        loss = -self.fit(X, y).sum()
        # print('loss', loss)
        self.zero_grad()
        loss.backward()
        # print('grad', X.grad.data.sign())
        return epsilon * X.grad.data.sign()
    
    def train_step(self, X, y, opt, epsilon):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        opt.zero_grad()
        delta = self.fgsm(X, y, epsilon)
        X_pert = X + delta
        loss = -self.fit(X_pert, y).sum()
        loss.backward()
        opt.step()
        # return {'loss': nll.item(), 'length': self.length_scale.detach().cpu(), 
        #         'noise': self.noise_scale.detach().cpu(), 
        #         'amplitude': self.amplitude_scale.detach().cpu()}

def mse_loss(output, target):
    # print('mse', target.size(), output.size())
    return torch.pow(torch.abs(target-output), 2).sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ndim', type=int)
    args = parser.parse_args()
    print('args: ', args)
    global BEST_MODEL_PATH, num_dim
    num_dim = args.ndim
    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0]
    BEST_MODEL_PATH = f'gpr_fgsm_{num_dim}d_gaussian.pt'
    global x_test
    x_test = torch.randn(TEST_SIZE, num_dim)
    e_test = torch.randn(TEST_SIZE)
    y_test = torch.unsqueeze(x_test.sum(1) + e_test, dim=1)
    x_test_mean_squared = torch.mean(x_test ** 2).item()
    print('x_test', x_test)
    print('y_test', y_test)

    test_losses = np.zeros((len(epsilons), TRAIN_SIZE))
    tic = process_time()
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        for train_size in range(1, TRAIN_SIZE+1):
            temp = np.zeros(N)
            for j in range(N):
                x_train = torch.randn(train_size, num_dim)
                e_train = torch.randn(train_size)
                y_train = torch.unsqueeze(x_train.sum(1) + e_train, dim=1)
                model = GP()
                opt = Adam(model.parameters(), lr=learning_rate)
                model.train()
                for epoch in range(num_epochs):
                    model.train_step(x_train, y_train, opt, epsilon)
                mu, var = model.forward(x_test)
                if (train_size == TRAIN_SIZE) and (j == N-1):
                    print('eps', epsilon)
                    print('mu', mu)
                    print('var', var)
                
                model.eval()
                test_loss = mse_loss(mu, y_test) / TEST_SIZE
                temp[j] = test_loss / x_test_mean_squared
            mean = np.mean(temp)
            test_losses[i, train_size-1] = mean
    toc = process_time()
    print("elapsed time:", toc - tic)
    print("test_losses:", test_losses)

if __name__ == "__main__":
    main()
