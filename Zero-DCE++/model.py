import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv(nn.Module):
  def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
    super(SeparableConv, self).__init__()

    self.depthwise = nn.Conv2d(c_in, c_in, kernel_size, stride, padding, groups=c_in, bias=True)
    self.pointwise = nn.Conv2d(c_in, c_out, 1, stride, padding=0, bias=True)

  def forward(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    return x

class DCE_Net_ext(nn.Module):
  def __init__(self, scale_factor=1.0):
    super(DCE_Net_ext, self).__init__()

    # Channels
    self.c_mid = 32
    self.c_out = 3
    self.scale_factor = scale_factor

    self.e_conv1 = SeparableConv(3, self.c_mid)
    self.e_conv2 = SeparableConv(self.c_mid, self.c_mid)
    self.e_conv3 = SeparableConv(self.c_mid, self.c_mid)
    self.e_conv4 = SeparableConv(self.c_mid, self.c_mid)
    self.e_conv5 = SeparableConv(self.c_mid * 2, self.c_mid)
    self.e_conv6 = SeparableConv(self.c_mid * 2, self.c_mid)
    self.e_conv7 = SeparableConv(self.c_mid * 2, self.c_out)

  def forward(self, x):
    if self.scale_factor > 1:
      x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False)
    else:
      x_down = x

    x1 = F.relu(self.e_conv1(x_down))
    x2 = F.relu(self.e_conv2(x1))
    x3 = F.relu(self.e_conv3(x2))
    x4 = F.relu(self.e_conv4(x3))
    x5 = F.relu(self.e_conv5(torch.cat((x4, x3), 1)))
    x6 = F.relu(self.e_conv6(torch.cat((x5, x2), 1)))
    A = F.tanh(self.e_conv7(torch.cat((x6, x1), 1)))

    if self.scale_factor > 1:
      A = F.interpolate(A, size=x.shape[2:], mode='bilinear', align_corners=False)

    for k in range(8):
        x = x + A * x * (1.0 - x)

    return x, A