from torch import nn
import torch


class Source(nn.Module):
    def __init__(self, type= "Gaussian Mixture",
                       num_gaussians= 1):
        super().__init__()
        self.type = type
        self.num_gaussians = num_gaussians

        self.weights = nn.Parameter(torch.rand(num_gaussians))
        self.means = nn.Parameter(torch.rand(num_gaussians))
        self.variances = nn.Parameter(torch.rand(num_gaussians))

    def forward(self):
        if self.type == "Gaussian Mixture":
            return self.gaussian_mixture()
        else:
            raise NotImplementedError

    def gaussian_mixture(self, XY_grid, weights, mu, sigma) -> torch.Tensor:

        prefac = torch.sqrt((2 * torch.pi) ** 2 * torch.abs(sigma.sum(dim=0)))

        inside = (XY_grid[:, None, None, :, :] - mu[:, :, :, None, None]) ** 2. / (2 * sigma[:, :, :, None, None])
        exponent = -1 * torch.sum(inside, dim=0)
        gaussian = torch.exp(exponent) / prefac[:, :, None, None]

        gaussian = gaussian * weights[:, :, None, None]
        gaussian = gaussian.sum(dim=1)

        gaussian = gaussian + self.bias

        return gaussian

