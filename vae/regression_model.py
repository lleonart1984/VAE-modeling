from torch import nn
from torch import Tensor
import vae.modeling


class BlockRegression(nn.Module, vae.modeling.RegressionModel):
    """"
    Simple regression model implementation with a block.
    """

    def __init__(self, input_dim, output_dim, width, depth, activation=nn.ReLU, loss_function=nn.MSELoss()):
        super(BlockRegression, self).__init__()
        self.model = vae.modeling.BlockModule(input_dim, output_dim, width, depth, activation)
        self.loss_function = loss_function

    def eval(self, x: Tensor) -> Tensor:
        return self.model(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def train_batch(self, batch):
        x, target = batch
        y = self.forward(x)
        loss = self.loss_function(y, target)
        loss.backward()
        return loss.item(),


