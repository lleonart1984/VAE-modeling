import matplotlib.pyplot as plt
import numpy as np
import vae.training
import vae.cvae_model
import vae.dataman
import torch
import torch.nn as nn


N = 1000000


def test_distribution(c):
    size = len(c)
    sel = (np.random.uniform(0, 1, (size,)) < 0.2).astype(np.float32)
    x1 = np.random.normal(0, 1, (size,)) * (np.abs(c) + 0.1) + c
    x2 = np.random.normal(0, 1, (size,)) * (np.abs(0.5 - c) + 0.2)*0.1 + 0.5 - c * 0.5
    return x1*(1 - sel) + x2*sel


c = np.random.uniform(-1, 1, (N,))
x = test_distribution(c)  # np.random.normal(0, 1, (N,)) * (np.abs(c) + 0.1) + c
data = {'c': c, 'x': x}
dataset = vae.dataman.DataManager({'c': None, 'x': None}, (1, 1))
dataset.load_data(data)

print('Loaded data... '+str(dataset.data.shape))


def factory(epochs, batch_size):
    print('Creating CVAE model...')
    model = vae.cvae_model.CVAEModel(1, 1, 1, 16, 3, activation=nn.LeakyReLU)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.000001)
    gamma = np.exp(np.log(0.02) / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return model, optimizer, scheduler


path = "./Running/test_cvae"
# vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, factory, batch_size=8*1024, epochs=400)
print('Training finished at epoch ...'+str(len(state.history)))


loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list, loss_history)


# testing model evaluation
c_constant = -0.4
c_test = torch.Tensor(N*10, 1).fill_(c_constant).to(vae.training.DEFAULT_DEVICE)
state.model.train(False)
y_test = state.model.conditional_sampling(c_test)
x_test = test_distribution(c_test.cpu().numpy().flatten())  # np.random.normal(0, 1, (N,)) * (np.abs(c_constant) + 0.1) + c_constant

plt.figure()
plt.hist(y_test.cpu().detach().numpy(), density=True, bins=80)
plt.hist(x_test, histtype='step', density=True, bins=80)

plt.show()

