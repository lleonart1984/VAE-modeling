import matplotlib.pyplot as plt
import numpy as np
import vae.training
import vae.cvae_model
import vae.regression_model
import vae.dataman
import torch
import torch.nn as nn


dataset = vae.dataman.DataManager(
    mappings={
        'sigma': None,
        'albedo': lambda a: np.power(1 - a, 1.0 / 6),
        'g': None,
        # 'logscat': None,
        'output_z': None,
        'output_b': None,
        'output_a': None
    },
    blocks=(3, 3)
)
dataset.load_file("./DataSets/SphereScattersDataSet.npz")  #, limit=1024*1024)
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
    z_pos = test_data[:, 3].cpu().numpy()
    plt.hist(z_pos, density=True, bins=80)


# test_dataset(16.0, 0.99, 0.0)
# plt.show()
# exit()


def cvae_factory(epochs, batch_size):
    print('Creating CVAE model...')
    model = vae.cvae_model.CVAEModel(3, 3, 4, 8, 4, activation=nn.LeakyReLU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
    gamma = np.exp(np.log(0.02) / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return model, optimizer, scheduler


path = "./Running/sphere_cvae"
# vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, cvae_factory, batch_size=128*1024, epochs=8000)
print('Training finished at epoch ...'+str(len(state.history)))


loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list, loss_history)

# testing model evaluation

dataset = vae.dataman.DataManager(
    mappings={
        'sigma': None,
        'albedo': lambda a: np.power(1 - a, 1.0 / 6),
        'g': None,
        'output_z': None,
        'output_b': None,
        'output_a': None,
        'logscat': None,
    },
    blocks=(3, 1, 3)
)
dataset.load_file("./DataSets/Test_SphereScattersDataSet.npz")
dataset.set_device(vae.training.DEFAULT_DEVICE)
print('Loaded data for testing... '+str(dataset.data.shape))
state.model.train(False)


def test_setting(sigma, albedo, g):
    test_data = dataset.get_filtered_data({
        'sigma': (sigma - .5, sigma + .5),
        'albedo': (albedo - 0.0001, albedo),
        'g': (g - 0.01, g + 0.01)
    })
    print('Testing data frame '+str(test_data.shape))
    if len(test_data) == 0:
        print(f'[ERROR] Config {sigma},{albedo},{g} has no data.')
        return

    plt.figure()

    # drawing histograms from empirical z position distribution
    weights = np.exp(test_data[:, 6].cpu().numpy())
    z_pos = test_data[:, 3].cpu().numpy()
    plt.hist(z_pos, weights=weights, density=True, bins=80)
    # plt.hist(z_pos, density=True, bins=80)

    # drawing histograms from model sampling distribution
    internal_albedo = np.power(1.0 - albedo, 1.0/6)
    sigma_test = torch.Tensor(10000, 1).fill_(sigma).to(vae.training.DEFAULT_DEVICE)
    albedo_test = torch.Tensor(10000, 1).fill_(internal_albedo).to(vae.training.DEFAULT_DEVICE)
    g_test = torch.Tensor(10000, 1).fill_(g).to(vae.training.DEFAULT_DEVICE)
    y_test = state.model.conditional_sampling(torch.cat([sigma_test, albedo_test, g_test], dim=1))
    plt.hist(torch.clamp(y_test[:, 0], -1, 1).cpu().detach().numpy(), density=True, bins=80, histtype='step')

    plt.show()


testing_sigmas = [1.0, 4.0, 9.0, 20.0]
testing_albedos = [0.95]
# testing_albedos = [1.0, 0.999, 0.95, 0.8]
testing_gs = [-0.5, 0.0, 0.7, 0.875]

for sigma in testing_sigmas:
    for albedo in testing_albedos:
        for g in testing_gs:
            test_setting(sigma, albedo, g)

# testing model evaluation
plt.show()

