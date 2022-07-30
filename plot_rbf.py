from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.datasets import make_moons

num_dim = 2
mu = torch.ones(num_dim)
sigma = torch.eye(num_dim)
x = torch.cat([torch.distributions.MultivariateNormal(-mu, sigma).sample((10,)), torch.distributions.MultivariateNormal(mu, sigma).sample((10,))]).float().numpy()
y = torch.cat([-torch.ones(10), torch.ones(10)]).float().numpy()
# x, y = make_moons(20, noise=0.1)
# y[np.where(y==0)] = -1
fig, ax = plt.subplots()
print('x', x)
print('y', y)
print('x[np.where(y==-1),0]', x[np.where(y==-1),0])
print('x[np.where(y==-1),1]', x[np.where(y==-1),1])
ax.scatter(x[np.where(y==-1),0], x[np.where(y==-1),1], label='Class 1')
ax.scatter(x[np.where(y==1),0], x[np.where(y==1),1], label='Class 2')
ax.set_title('Training data')
ax.legend()
plt.savefig('plot_rbf.png')
