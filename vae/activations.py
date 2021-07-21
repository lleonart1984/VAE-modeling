import torch.nn as nn
import torch.functional as F


class ShiftedReLU(nn.Module):

    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return F.relu(x + self.offset)


class ShiftedSoftplus(nn.Module):

    def __init__(self, offset=1):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return F.softplus(x - self.offset)
