import matplotlib.pyplot as plt
import numpy as np
import vae.training
import vae.cvae_model
import vae.dataman
import torch
import torch.nn as nn


def test_distribution(c):
    pass

N = 10000
c = np.random.uniform(-1, 1, (N,))
x = np.random.normal(0, 1, (N,)) * (np.abs(c) + 0.1) + c
data = {'c': c, 'x': x}
dataset = vae.dataman.DataManager({'c': None, 'x': None}, (1, 1))
dataset.load_data(data)

print('Loaded data... '+str(dataset.data.shape))


def factory(epochs, batch_size):
    print('Creating CVAE model...')
    model = vae.cvae_model.CVAEModel(1, 1, 1, 20, 2, activation=nn.ELU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    return model, optimizer, scheduler


path = "./Running/test_cvae"
# vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, factory, epochs=400)
print('Training finished at epoch ...'+str(len(state.history)))


loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list, loss_history)


# testing model evaluation
c_constant = 0.01
c_test = torch.Tensor(N, 1).fill_(c_constant).to(vae.training.DEFAULT_DEVICE)
state.model.train(False)
y_test = state.model.conditional_sampling(c_test)
x_test = np.random.normal(0, 1, (N,)) * (np.abs(c_constant) + 0.1) + c_constant

plt.figure()
plt.hist(y_test.cpu().detach().numpy(), density=True, bins=40)
plt.hist(x_test, histtype='step', density=True, bins=40)

plt.show()

