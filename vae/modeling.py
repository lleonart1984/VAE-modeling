import torch
from torch import nn
from torch import Tensor
from typing import List


class BlockModule(nn.Module):
    """
    Block Model. Represents a simple model with d layers of w nodes and input_dim, output_dim as first and last
    layer size.
    Activations differs if they are after inner layers or last layer.
    """

    def __init__(self, input_dim, output_dim, w, d, activation=nn.LeakyReLU, last_activation=None):
        """
        Creates a NN with <D> hidden layers of <W> nodes each.
        Input layer has size <input_dim> and output layer <output_dim>.
        All dense layers are activated with <activation> except last one that uses linear.
        """
        super(BlockModule, self).__init__()
        modules = [nn.Sequential(_dense(input_dim, w if d > 0 else output_dim),
                                 activation() if d > 0 else (last_activation() if last_activation is not None else None))]
        # first hidden layer
        # next depths - 1 hidden layers
        if d > 0:
            for _ in range(d - 1):
                modules.append(nn.Sequential(_dense(w, w), activation()))
            # last layer
            if last_activation is not None:
                modules.append(nn.Sequential(_dense(w, output_dim), last_activation()))
            else:
                modules.append(_dense(w, output_dim))  # last layer has linear activation
        self.model = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def _dense(a, b):
    """
    Tool method to create a layer of b nodes fully connected with previous a nodes.
    The initialization is xavier uniform
    """
    d = nn.Linear(a, b)
    torch.nn.init.xavier_uniform_(d.weight)
    # torch.nn.init.kaiming_normal_(d.weight)
    torch.nn.init.uniform_(d.bias, -0.1, 0.1)
    return d


# Represents a customized conditional probabilistic module with gaussians generation y~N(mu(x | c), e^logVar(x | c))
class GaussianModule(nn.Module):
    def __init__(self,
                 condition_dim, target_dim, output_dim,
                 width, depth, activation=nn.LeakyReLU, forced_log_var=None
                 ):
        """
        Creates the conditional model using block sequence of layers.
        if forced_logVar is not None the model generates mu and a constant logVar equals to that value.
        if forced_logVar is None the model generates both mu and logVar.
        """
        super(GaussianModule, self).__init__()
        self.output_dim = output_dim
        self.forced_log_var = forced_log_var
        self.condition_dim = condition_dim
        self.model = BlockModule(
            condition_dim + target_dim,
            output_dim * 2 if forced_log_var is None else output_dim,
            width, depth, activation
        )

    def forward(self, c: Tensor, x: Tensor) -> List[Tensor]:
        """
        Evaluates the model and return two tensors, one with mu and another with log variance
        """
        o = self.model(torch.cat([c, x], dim=-1))
        if self.forced_log_var is None:
            mu, log_var = o.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, -16, 16)
        else:
            mu, log_var = o, torch.full_like(o, self.forced_log_var)
        return [mu, log_var]

    @staticmethod
    def sample(mu: Tensor, log_var: Tensor) -> Tensor:
        return mu + torch.exp(log_var * 0.5) * torch.randn_like(mu)

    @staticmethod
    def log_likelihood(x: Tensor, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Compute the posterior log-likelihood of x assuming N(xMu, xVar)
        1.837877066 == log (2 * pi)
        Norm(x) = exp(-0.5 (x - xMu)**2 / s**2 ) / (s**2 * 2 * pi)^0.5
        log-Norm = -0.5 ((x - xMu)**2 / s**2 + log(s**2) + log(2 * pi))
        """
        return torch.sum((x - mu) ** 2 / torch.exp(log_var) + log_var + 1.837877066, dim=-1, keepdim=True).float()*-0.5

    @staticmethod
    def standard_log_likelihood(x: Tensor) -> Tensor:
        """
        Compute the posterior log-likelihood of x assuming N(0, I)
        1.837877066 == log (2 * pi)
        Norm(x) = exp(-0.5 (x - xMu)**2 / s**2 ) / (s**2 * 2 * pi)^0.5
        log-Norm = -0.5 ((x - xMu)**2 / s**2 + log(s**2) + log(2 * pi))
        """
        return torch.sum(x ** 2 + 1.837877066, dim=-1, keepdim=True).float() * -0.5

    @staticmethod
    def kl_divergence(mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Compute KL Divergence between prior distribution and the standard normal distribution
        """
        return 0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1 - log_var, dim=-1, keepdim=True).float()


class NNModel:
    def train_batch(self, batch):
        pass


class RegressionModel(NNModel):
    def eval(self, x):
        pass


class GenerativeModel(NNModel):
    def sampling(self, samples):
        pass


class ConditionalGenerativeModel(NNModel):
    def conditional_sampling(self, c: Tensor):
        pass

