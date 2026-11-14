import torch
import torch.nn as nn
from torch.nn import functional as F

class DCE_Net(nn.Module):
  def __init__(self):
    super(DCE_Net, self).__init__()

    # Channels
    self.c_mid = 32
    self.c_out = 24

    self.e_conv1 = nn.Conv2d(3, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv2 = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv3 = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv4 = nn.Conv2d(self.c_mid, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv5 = nn.Conv2d(self.c_mid * 2, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv6 = nn.Conv2d(self.c_mid * 2, self.c_mid, 3, 1, 1, bias=True)
    self.e_conv7 = nn.Conv2d(self.c_mid * 2, self.c_out, 3, 1, 1, bias=True)

  def forward(self, x):
    x1 = F.relu(self.e_conv1(x))
    x2 = F.relu(self.e_conv2(x1))
    x3 = F.relu(self.e_conv3(x2))
    x4 = F.relu(self.e_conv4(x3))
    x5 = F.relu(self.e_conv5(torch.cat((x4, x3), 1)))
    x6 = F.relu(self.e_conv6(torch.cat((x5, x2), 1)))
    A = F.tanh(self.e_conv7(torch.cat((x6, x1), 1)))

    A_split = torch.split(A, 3, dim=1)
    for k in range(8):
        x = x + A_split[k] * x * (1.0 - x)

    return x, A