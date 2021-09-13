import matplotlib.pyplot as plt
import numpy as np
import vae.training
import vae.cvae_model
import vae.regression_model
import vae.dataman
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = vae.dataman.DataManager(
    mappings={
        'sigma': None,
        'albedo': lambda a: np.power(1 - a, 1.0 / 6),
        'g': None,
        'logscat': None,
    },
    blocks=(3, 1)
)
dataset.load_file("./DataSets/SphereScattersDataSet_Abs.npz")  #, limit=1024*1024)
print('Loaded data... '+str(dataset.data.shape))


def test_dataset(sigma, albedo, g):
    test_data = dataset.get_filtered_data({
        'sigma': (sigma - 1, sigma + 2),
        'albedo': (albedo - 0.001, albedo),
        'g': (g - 0.2, g + 0.2)
    })
    if len(test_data) == 0:
        print(f'[ERROR] Config {sigma},{albedo},{g} has no data.')
        return
    print(len(test_data))
    plt.figure()
    # drawing histograms from empirical z position distribution
    scat = np.exp(test_data[:, 3].cpu().numpy())
    mean = np.mean(scat)
    print(mean)
    plt.hist(scat, density=True, bins=80)


# test_dataset(4.0, 0.9999, 0.9)
# plt.show()
# exit()


class RegLoss(nn.Module):

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, x, y):
        return F.mse_loss(torch.exp(x), torch.exp(y))


def reg_factory(epochs, batch_size):
    print('Creating Regression model...')
    model = vae.regression_model.BlockRegression(3, 1, 8, 4, activation=nn.LeakyReLU, loss_function=RegLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
    gamma = np.exp(np.log(0.01) / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return model, optimizer, scheduler


path = "./Running/sphere_reg"

# vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, reg_factory, batch_size=256*1024, epochs=1000)
print('Training finished at epoch ...'+str(len(state.history)))

loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list[-500:], loss_history[-500:])
plt.show()


def test_setting(albedo):
    test_data = dataset.get_filtered_data({
        'albedo': (albedo - 0.001, albedo),
    })
    plt.figure()
    # drawing histograms from empirical z position distribution
    weights = np.exp(test_data[:, 3].cpu().numpy())
    d_values = test_data[:, 0].cpu().numpy()
    g_values = test_data[:, 2].cpu().numpy()
    plt.hist2d(x=d_values, y=g_values, weights=weights, bins=50)
    # plt.hist(z_pos, density=True, bins=80)

    plt.show()


test_setting(1.00)
test_setting(.999)
test_setting(.99)
test_setting(.9)
