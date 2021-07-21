import matplotlib.pyplot as plt
import numpy as np
import vae.training
import vae.regression_model
import vae.dataman
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


N = 10000
x = np.random.uniform(-1, 1, (N,))
y = np.sin(x*10)*(x**2)+x
data = {'x': x, 'y': y}
dataset = vae.dataman.DataManager({'x': None, 'y': None}, (1, 1))
dataset.load_data(data)

print('Loaded data... '+str(dataset.data.shape))


def factory(epochs, batch_size):
    print('Creating model...')
    model = vae.regression_model.BlockRegression(1, 1, 16, 2, activation=nn.ELU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    return model, optimizer, scheduler


path = "./Running/test_regression"
vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, factory, epochs=400)
print('Training finished at epoch ...'+str(len(state.history)))


loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list, loss_history)


# testing model evaluation
x_test = torch.tensor(np.random.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)).to(vae.training.DEFAULT_DEVICE)
state.model.train(False)
y_test = state.model.eval(x_test)

plt.figure()
plt.scatter(x, y)
plt.scatter(x_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())

plt.show()








