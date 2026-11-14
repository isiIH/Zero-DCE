import torch
from torch import nn
from torch.nn import functional as F

class SPA_Loss(nn.Module):
  def __init__(self, weight=1.0):
    super(SPA_Loss, self).__init__()

    self.weight = weight

    kernel = torch.tensor([
        [[[0,  0, 0], [-1,  1,  0], [0,  0, 0]]],  # Left
        [[[0,  0, 0], [ 0,  1, -1], [0,  0, 0]]],  # Right
        [[[0, -1, 0], [ 0,  1,  0], [0,  0, 0]]],  # Up
        [[[0,  0, 0], [ 0,  1,  0], [0, -1, 0]]]   # Down
    ], dtype=torch.float32)
    self.register_buffer('kernel', kernel, persistent=False)
    self.pool = nn.AvgPool2d(4)

  def forward(self, org_img, pred_img):
    # Y
    org_mean = torch.mean(org_img, 1, keepdim=True)
    org_mean = self.pool(org_mean)

    # I
    pred_mean = torch.mean(pred_img, 1, keepdim=True)
    pred_mean = self.pool(pred_mean)

    # (B, 4, H/4, W/4)
    d_org = F.conv2d(org_mean, self.kernel, padding=1)
    d_enhance = F.conv2d(pred_mean, self.kernel, padding=1)

    diff = torch.pow(d_org - d_enhance, 2)

    return self.weight * torch.mean(torch.sum(diff, dim=1))

class EXP_Loss(nn.Module):
  def __init__(self, weight=1.0):
    super(EXP_Loss, self).__init__()

    self.weight = weight

    self.E = 0.6
    self.pool = nn.AvgPool2d(16)

  def forward(self, x):
    Y = torch.mean(x, 1, keepdim=True)
    Y = self.pool(Y)

    return self.weight * torch.mean(torch.pow((Y - self.E), 2))

class COL_Loss(nn.Module):
  def __init__(self, weight=1.0):
    super(COL_Loss, self).__init__()

    self.weight = weight

  def forward(self, x):
    J = torch.mean(x, [2,3])
    r, g, b = torch.split(J, 1, dim=1)

    d_rg = torch.pow(r - g, 2)
    d_rb = torch.pow(r - b, 2)
    d_gb = torch.pow(g - b, 2)

    return self.weight * torch.mean(d_rg + d_rb + d_gb)

class ILLUM_Loss(nn.Module):
  def __init__(self, weight=1.0):
    super(ILLUM_Loss, self).__init__()
    self.weight = weight

  def forward(self, x):
    dx = torch.diff(x, dim=2) # H
    dy = torch.diff(x, dim=3) # W

    diff_h_sq = dx ** 2
    diff_w_sq = dy ** 2

    tv_h = torch.mean(diff_h_sq, dim=[2, 3])
    tv_w = torch.mean(diff_w_sq, dim=[2, 3])

    return self.weight * 2 * torch.mean(torch.sum(tv_h + tv_w, dim=1))
