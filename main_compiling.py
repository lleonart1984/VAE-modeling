import torch
import torch.nn as nn
import numpy as np
import vae.training
import vae.cvae_model
import vae.compiling


def cvae_factory(epochs, batch_size):
    print('Creating CVAE model...')
    model = vae.cvae_model.CVAEModel(5, 4, 8, 16, 6, activation=lambda: nn.LeakyReLU(negative_slope=0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0000001)
    gamma = np.exp(np.log(0.001) / epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return model, optimizer, scheduler

# path = "./Running/linear_sphere_cvae"
path = "./Running/linear_sphere_cvae_na"
model: vae.cvae_model.CVAEModel = vae.training.get_last_model(path, cvae_factory, top_epoch=None)

compiler = vae.compiling.HLSLCompiler()

code = compiler.compile({
    'naPathModel': model.decoder.model.model
})

print(code)