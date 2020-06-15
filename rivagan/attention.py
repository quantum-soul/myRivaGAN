import torch
from torch import nn
from torch.nn import functional


def multiplicative(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the
    batch, the data vector is combined with the first D dimensions of the 5d
    tensor through an elementwise product.

    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in}, L, H, W)
    """
    # N is the batch size, L is the sequence length
    N, D = data.size()
    N, C, L, H, W = x.size()
    assert D <= C, "data dims must be less than channel dims"
    x = torch.cat([
        x[:, :D, :, :, :] * data.view(N, D, 1, 1, 1).expand(N, D, L, H, W),
        x[:, D:, :, :, :]
    ], dim=1)
    return x


class AttentiveEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D, )
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, tie_rgb=False, linf_max=0.016,
                 kernel_size=(1, 11, 11), padding=(0, 5, 5)):
        super(AttentiveEncoder, self).__init__()

        self.linf_max = linf_max
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.padding = padding

        self._attention = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, data_dim, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(data_dim),
        )
        self._conv = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 1 if tie_rgb else 3, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = functional.softmax(self._attention(frames), dim=1)
        x = torch.sum(multiplicative(x, data), dim=1, keepdim=True)
        x = self._conv(torch.cat([frames, x], dim=1))
        return frames + self.linf_max * x


class AttentiveDecoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, encoder):
        super(AttentiveDecoder, self).__init__()
        self.data_dim = encoder.data_dim
        self._attention = encoder._attention
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size, padding=encoder.padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, self.data_dim, kernel_size=encoder.kernel_size,
                      padding=encoder.padding, stride=1),
        )

    def forward(self, frames):
        N, D, L, H, W = frames.size()
        attention = functional.softmax(self._attention(frames), dim=1)
        x = self._conv(frames) * attention
        # mean()函数的参数：返回输入张量给定维度dim上每行的均值。dim=0,按行求平均值，返回的形状是（1，列数）；
        # dim=1,按列求平均值，返回的形状是（行数，1）,默认不设置dim的时候，返回的是所有元素的平均值。
        # 这里相当于返回D个（W,H）的均值
        return torch.mean(x.view(N, self.data_dim, -1), dim=2)
