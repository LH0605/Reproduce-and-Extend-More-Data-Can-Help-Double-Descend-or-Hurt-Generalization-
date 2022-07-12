import numpy as np
import matplotlib.pyplot as plt

# epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0] # linear loss
epsilons = [0, 0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.3, 1.5, 1.7, 2.0] # gaussian
# epsilons = [0, 1., 2., 3., 4., 5., 6., 8., 10., 12.] # poisson

test_losses = ???
TRAIN_SIZE = len(test_losses)
train_sizes = np.arange(1, TRAIN_SIZE+1)

plt.title("Linear Regression Gaussian")
plt.xlabel("Size of Training Dataset")
plt.ylabel("Test Loss")
for i in range(len(epsilons)): # range(len(epsilons)):
    print('eps:', epsilons[i])
    plt.plot(train_sizes, test_losses[i], label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="best")
plt.savefig(f"linreg_gaussian_fgsm_1d_all.png")
plt.clf()

"""
# 1
plt.title("Linear Regression Gaussian")
plt.xlabel("Size of Training Dataset")
plt.ylabel("Test Loss")
for i in [1,2,3]: # range(len(epsilons)):
    print('eps:', epsilons[i])
    plt.plot(train_sizes, test_losses[i], label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="best")
plt.savefig(f"linreg_gaussian_fgsm_1d_weak.png")
plt.clf()

# 2
for i in [13, 14, 15, 16]:
    print('eps:', epsilons[i])
    plt.plot(train_sizes, test_losses[i], label=f"Ɛ = {epsilons[i]}")
plt.legend(loc="best")
plt.savefig(f"linreg_gaussian_fgsm_1d_strong.png")
"""
