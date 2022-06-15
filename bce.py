import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# FGSM attack
def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
#     print('fgsm X, delta, y, sum:', X.size(), delta.size(), y.size(), X.float() + delta)
    y_pred = model(X + delta)
    loss = nn.CrossEntropyLoss()(y_pred,y)
#     print('fgsm loss:', loss)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

w = torch.tensor(1., requires_grad=True)

def epoch_adversarial(loader, model, loss_fun, opt, epsilon):
#     total_loss = 0.
    for X, y in loader:
        delta = fgsm(model, X.float(), y.long(), epsilon)
        # perturbed training data
        x_pert = X.float() + delta
        # predicted output
        y_pred = model(x_pert)
        
        # clear gradients wrt to parameters
        opt.zero_grad()
        # calculate linear loss
        loss = loss_fun(y_pred, y.long())
        # get gradients wrt to parameters
        loss.backward()

        opt.step()

def test_loss(loader, model, loss_fun):
    total_loss = 0.
    for X, y in loader:
        y_pred = model(X.float())
        test_loss = loss_fun(y_pred, y.long())
        total_loss += test_loss.item() * X.shape[0]
    return total_loss / len(loader.dataset)

def main():
    mu = 1
    sigma = 2.0
    learning_rate = 1e-3
    epsilon = 1.2
    epochs = 1000
    TRAIN_SIZE = 30
    TEST_SIZE = 50

    # Loss
    loss_fun = nn.CrossEntropyLoss()

    test_losses = np.zeros(TRAIN_SIZE)

    # train the model
    for train_size in range(1, TRAIN_SIZE+1):
    #     train_size = 2
        batch_size = train_size
        # w = torch.tensor(1., requires_grad=True)
        model = torch.nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
        )
        
        # define optimizer
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        x_train = torch.unsqueeze(torch.linspace(-10, 10, 2*train_size), dim=1)
        y_train = torch.cat([torch.zeros(train_size), torch.ones(train_size)])
        x_test = torch.unsqueeze(torch.linspace(-10, 10, 2*TEST_SIZE), dim=1)
        y_test = torch.cat([torch.zeros(TEST_SIZE), torch.ones(TEST_SIZE)])

        # define datasets and data loaders
        train_set = Data.TensorDataset(x_train, y_train)
        train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_set = Data.TensorDataset(x_test, y_test)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_adversarial(train_loader, model, loss_fun, opt, epsilon)

        test_losses[train_size-1] = test_loss(test_loader, model, loss_fun)

    # logging
    # print('test_losses', test_losses)
    train_sizes = np.arange(1, TRAIN_SIZE+1)
    plt.plot(train_sizes, test_losses, label=f"Ɛ = {epsilon}")
    plt.xlabel("Size of Training Dataset")
    plt.ylabel("Test Loss")
    plt.savefig("bce.png")
    plt.clf()


if __name__ == "__main__":
    main()
