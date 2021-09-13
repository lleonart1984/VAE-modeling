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
        'sigmax': None,
        'sigmay': None,
        'sigmaz': None,
        'albedo': lambda a: np.power(1 - a, 1.0 / 6),
        'g': None,
        # 'logscat': None,
        'output_sx_x': None,
        'output_sx_y': None,
        'output_sw_x': None,
        'output_sw_y': None
    },
    blocks=(6, 4)
)
dataset.load_file("./DataSets/LinearSphereScattersDataSet.npz")  #, limit=1024*1024)
print('Loaded data... '+str(dataset.data.shape))


def test_dataset(sigma, sigmax, sigmay, sigmaz, albedo, g):
    test_data = dataset.get_filtered_data({
        'sigma': (sigma - 1, sigma + 2),
        'sigmax': (sigmax - sigma*0.2, sigmax + sigma*0.2),
        'sigmay': (sigmay - sigma*0.2, sigmay + sigma*0.2),
        'sigmaz': (sigmaz - sigma*0.2, sigmaz + sigma*0.2),
        # 'albedo': (albedo - 0.001, albedo),
        #'g': (g - 0.2, g + 0.2)
    })
    if len(test_data) == 0:
        print(f'[ERROR] Config {sigma},{albedo},{g} has no data.')
        return
    print(len(test_data))
    plt.figure()
    # drawing histograms from empirical z position distribution
    sx_x = test_data[:, 6].cpu().numpy()
    sx_y = test_data[:, 7].cpu().numpy()
    plt.hist2d(sx_x, sx_y, density=True, bins=80)


# test_dataset(16.0, .0, .0, -16.0, 0.99, 0.0)
# plt.show()
# exit()


def cvae_factory(epochs, batch_size):
    print('Creating CVAE model...')
    model = vae.cvae_model.CVAEModel(6, 4, 8, 16, 5, activation=nn.LeakyReLU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
    gamma = np.exp(np.log(0.005) / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return model, optimizer, scheduler


path = "./Running/linear_sphere_cvae"
# vae.training.clear_training(path)  # comment if you want to reuse from previous training
state = vae.training.start_training(path, dataset, cvae_factory, batch_size=32*1024, epochs=1000)
print('Training finished at epoch ...'+str(len(state.history)))


loss_history = np.array([h[0] for h in state.history])
epoch_list = np.arange(0, len(loss_history), 1)

plt.figure()
plt.plot(epoch_list, loss_history)

# testing model evaluation

dataset = vae.dataman.DataManager(
    mappings={
        'sigma': None,
        'sigmax': None,
        'sigmay': None,
        'sigmaz': None,
        'albedo': lambda a: np.power(np.maximum(1 - a, 0.0), 1.0 / 6),
        'g': None,
        'output_sx_x': None,
        'output_sx_y': None,
        'output_sw_x': None,
        'output_sw_y': None,
        'logscat': None,
    },
    blocks=(6, 4, 1)
)
dataset.load_file("./DataSets/Test_LinearSphereScattersDataSet.npz")
dataset.set_device(vae.training.DEFAULT_DEVICE)
print('Loaded data for testing... '+str(dataset.data.shape))
state.model.train(False)


def test_setting(sigma, gradient, albedo, g):
    test_data = dataset.get_filtered_data({
        'sigma': (sigma - .5, sigma + .5),
        'sigmax': (gradient[0] - .05, gradient[0] + .05),
        'sigmay': (gradient[1] - .05, gradient[1] + .05),
        'sigmaz': (gradient[2] - .05, gradient[2] + .05),
        'albedo': (albedo - 0.0005, albedo + 0.00001),
        'g': (g - 0.01, g + 0.01)
    })
    print('Testing data frame '+str(test_data.shape))
    if len(test_data) == 0:
        print(f'[ERROR] Config {sigma},{albedo},{g} has no data.')
        return

    _, slots = plt.subplots(1, 2, figsize=(8, 4))
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, h_pad=0, w_pad=0)
    plt.subplot(slots[0])
    # drawing histograms from empirical z position distribution
    weights = np.exp(test_data[:, 10].cpu().numpy())
    sx_x = test_data[:, 6].cpu().numpy()
    sx_y = test_data[:, 7].cpu().numpy()
    plt.hist2d(sx_x, sx_y, weights=weights, density=True, bins=80, range=[(-3.2, 3.2), (-3.2, 3.2)])
    # plt.hist(z_pos, density=True, bins=80)
    plt.yticks([])
    plt.xticks([])
    plt.subplot(slots[1])
    # drawing histograms from model sampling distribution
    internal_albedo = np.power(1.0 - albedo, 1.0/6)
    sigma_test = torch.Tensor(10000, 1).fill_(sigma).to(vae.training.DEFAULT_DEVICE)
    sigmax_test = torch.Tensor(10000, 1).fill_(gradient[0]).to(vae.training.DEFAULT_DEVICE)
    sigmay_test = torch.Tensor(10000, 1).fill_(gradient[1]).to(vae.training.DEFAULT_DEVICE)
    sigmaz_test = torch.Tensor(10000, 1).fill_(gradient[2]).to(vae.training.DEFAULT_DEVICE)
    albedo_test = torch.Tensor(10000, 1).fill_(internal_albedo).to(vae.training.DEFAULT_DEVICE)
    g_test = torch.Tensor(10000, 1).fill_(g).to(vae.training.DEFAULT_DEVICE)
    sx_test = torch.clamp(state.model.conditional_sampling(
        torch.cat(
            [sigma_test, sigmax_test, sigmay_test, sigmaz_test, albedo_test, g_test],
            dim=1
        )
    ), -3.14159, 3.14159).cpu().detach().numpy()
    plt.hist2d(sx_test[:, 0], sx_test[:, 1], density=True, bins=80, range=[(-3.2, 3.2), (-3.2, 3.2)])

    plt.show()


testing_sigmas = [1.0, 4.0, 9.0, 20.0]
testing_albedos = [0.95]
# testing_albedos = [1.0, 0.999, 0.95, 0.8]
testing_gs = [-0.5, 0.0, 0.7, 0.875]

for sigma in testing_sigmas:
    for albedo in testing_albedos:
        for g in testing_gs:
            scale = sigma * 1
            gradient = (1*scale, 0*scale, 0*scale)
            test_setting(sigma, gradient, albedo, g)

# testing model evaluation
plt.show()

