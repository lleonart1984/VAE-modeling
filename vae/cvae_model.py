import torch
from torch import nn
from torch import Tensor
from typing import List
import vae.modeling


class CVAEModel(nn.Module, vae.modeling.ConditionalGenerativeModel):
    """
    Conditional variational auto-encoder. Uses two probabilistic models to infer latent representation and target
    distribution.
    """

    P_MODEL = vae.modeling.GaussianModule

    def __init__(
            self, condition_dim, target_dim, latent_dim,
            width, depth, activation=nn.ReLU, forced_log_var=None
    ):
        super(CVAEModel, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        # The encoder triples the width nodes and increase the depth to be more expressive.
        # More complexity of the encoder doesnt affect the performance of the decoder.
        self.encoder = self.P_MODEL(
            condition_dim, target_dim, latent_dim,
            width * 3, depth + 1,
            activation=nn.Softplus
        )
        self.decoder = self.P_MODEL(
            condition_dim, latent_dim, target_dim,
            width, depth,
            activation=activation, forced_log_var=forced_log_var
        )

    def forward(self, c: Tensor, x: Tensor) -> List[Tensor]:
        z_mu, z_log_var = self.encoder(c, x)
        z = self.P_MODEL.sample(z_mu, z_log_var)
        x_mu, x_log_var = self.decoder(c, z)
        return [z_mu, z_log_var, z, x_mu, x_log_var]

    def conditional_sampling(self, c: Tensor) -> Tensor:
        z = torch.randn((len(c), self.latent_dim)).to(c.device)
        x_mu, x_log_var = self.decoder(c, z)
        return self.P_MODEL.sample(x_mu, x_log_var)

    def train_batch(self, batch):
        c, x = batch
        z_mu, z_log_var, z, x_mu, x_log_var = self.forward(c, x)
        posterior_ll = torch.mean(self.P_MODEL.log_likelihood(x, x_mu, x_log_var))
        kl_div = torch.mean(self.P_MODEL.kl_divergence(z_mu, z_log_var))
        elbo = posterior_ll - kl_div
        loss = -elbo
        # perform optimization
        loss.backward()
        return loss.item(), kl_div.item()
