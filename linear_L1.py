import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import SGD, Adam
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
mu = 1
sigma = 2
learning_rate = 1e-1
# epsilon = 1.5
num_epochs = 100
W = 1.
TEST_SIZE = 1000
TRAIN_SIZE = 20
# BEST_MODEL_PATH = 'best_gaussian_mixture_linear_loss_model.pt'
epsilons = [0, 0.3, 0.5, 0.7, 1.1, 1.3]

class WeightClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = w.clamp(-W, W)


clipper = WeightClipper()

def def_data(size):

    x = torch.cat([torch.distributions.Normal(-mu, sigma).sample((size,)),
                                         torch.distributions.Normal(mu, sigma).sample((size,))]).float()
    y = torch.cat([-torch.ones(size), torch.ones(size)]).float()

    # another group data generation
    # x, y = make_blobs(n_samples= size, n_features= 1,
    #                        centers=[np.full((1),-1) ,np.full((1),1)],cluster_std= 2)
    # y[y==0] = -1
    # return torch.FloatTensor(x), torch.FloatTensor(y)
    return x, y



def linear_loss(output, target):
    return torch.mul(-output, target)


# def fgsm(model, x, y, loss_fn, epsilon):
#     """ Construct FGSM adversarial examples on the examples X"""
#     x.requires_grad = True
#     output = model(x)
#     loss = loss_fn(output, y)
#     model.zero_grad()
#     loss.backward()
#     return epsilon * x.grad.data.sign()



def fit_train(train_loader, model, loss_fn,opt, epsilon):


    sum_loss = 0
    for x, y in train_loader:

        y_pred = model(x.view(x.shape[0], -1))[:,0] - y*epsilon*model.weight.norm(1)
        loss = torch.mean(loss_fn(y_pred, y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        # model.apply(clipper)
        sum_loss += loss.item() * x.shape[0]
        temp_train = sum_loss  / len(train_loader.dataset)
    return temp_train

def fit_test(test_loader,model, loss_fn, epsilon):
    sum_test_loss = 0
    for x, y in test_loader:
        # y_pred = model(x)
        # opt.zero_grad()
        # delta_test = fgsm(model, x, y, loss_fn, epsilon)
        y_pred = model(x.view(x.shape[0], -1))[:, 0] - y * epsilon * model.weight.norm(1)
        loss = torch.mean(loss_fn(y_pred, y))
        sum_test_loss += loss.item() * x.shape[0]
        temp_test = sum_test_loss / TEST_SIZE
    return temp_test


def main():
    loss_fn = linear_loss
    N = 100
    points_step_size = 1
    train_points = list(range(1, TRAIN_SIZE + 1))

    test_losses = np.zeros([len(epsilons), len(train_points), N])

    print(test_losses)
    model = nn.Linear(1, 1, bias=False)
    opt = Adam(model.parameters(), lr=learning_rate)
    for e_n in range(len(epsilons)):
        epsilon = epsilons[e_n]
        print('running training loop for epsilon = {}'.format(epsilon))
        for train_size in train_points:


            for j in range(1,N,1):
                model.reset_parameters()

                x_train, y_train = def_data(train_size)
                train_set = []
                # train_set = Data.TensorDataset(x_train, y_train)
                # train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
                for i in range(len(x_train)):
                    train_set.append([x_train[i], y_train[i]])
                train_loader = Data.DataLoader(train_set, shuffle=True, batch_size=train_size)


                for i in range(num_epochs):
                    train_loss = fit_train(train_loader, model, loss_fn, opt, epsilon,)
                    if i % clipper.frequency == 0:
                        model.apply(clipper)

                # train_loader = Data.DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)

                x_test, y_test = def_data(TEST_SIZE)
                test_set = []
                for i in range(len(x_test)):
                    test_set.append([x_test[i], y_test[i]])
                    test_loader = Data.DataLoader(test_set, shuffle=True, batch_size=x_test.size()[0])
                test_loss = fit_test(test_loader , model, loss_fn,  epsilon)


                test_losses[e_n, (train_size-1)// points_step_size, 0] = int(train_size)
                test_losses[e_n, (train_size-1)// points_step_size, j] = test_loss

                # print("test_losses:", test_losses)

    return test_losses


def plot_class_loss(epsilons,test_losses, training_points=list(range(1, TRAIN_SIZE+1))):
    x = training_points

    for eps_idx in range(len(epsilons)):
        eps = epsilons[eps_idx]
        test_eps = np.zeros(len(x))

        for n in range(len(x)):
            n = int(n)
            test_eps[n - 1] = np.mean(test_losses[eps_idx, n - 1, 1:])

        # ysmoothed = gaussian_filter1d(test_eps, sigma=2)
        plt.plot(x, test_eps, label="$\epsilon$ = {}".format(eps))
    plt.legend(loc='upper right')
    plt.xlabel('Size of training dataset')
    plt.ylabel('Test loss')
    plt.show()

plot_class_loss(epsilons, main())
# plot_class_loss(eps_list, train_classification_model(eps_list))
    # step = 3
    # train_sizes = np.arange(1, TRAIN_SIZE + 1)
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
    # for i in range(len(epsilons[1 + step:1 + (2 * step)])):
    #     epsilon = epsilons[1 + step + i]
    #     plt.plot(train_sizes, test_losses[1 + step + i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"linear_medium.png")
    #
    # plt.show()
    #
    # plt.title("Gaussian Mixture with Linear Loss (strong)")
    # plt.xlabel("Size of Training Dataset")
    # plt.ylabel("Test Loss")
    # plt.plot(train_sizes, test_losses[0], 'r--', label=f"Ɛ = 0")
    # for i in range(len(epsilons[1+(2*step):])):
    #     epsilon = epsilons[1+(2*step)+i]
    #     plt.plot(train_sizes, test_losses[1+(2*step)+i], label=f"Ɛ = {epsilon}")
    # plt.legend(loc="best")
    # plt.savefig(f"linear_strong.png")


# if __name__ == "__main__":
#     main()
