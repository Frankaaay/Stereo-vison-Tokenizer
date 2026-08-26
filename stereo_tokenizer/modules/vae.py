import torch
from dataclasses import dataclass
from einops import rearrange

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                reduction_dims = tuple(range(1, self.mean.ndim))
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=reduction_dims)
            else:
                reduction_dims = tuple(range(1, self.mean.ndim))
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=reduction_dims)

    def mode(self):
        return self.mean


@dataclass
class StructuredDiagonalGaussianPosterior:
    """VAE posterior that preserves the explicit batch/view axes."""

    distribution: DiagonalGaussianDistribution
    batch_size: int
    views: int

    def _structured(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.batch_size * self.views:
            raise RuntimeError("posterior batch no longer matches batch/view contract")
        return rearrange(
            tensor,
            "(b v) c t h w -> b v c t h w",
            b=self.batch_size,
            v=self.views,
        )

    @property
    def mean(self) -> torch.Tensor:
        return self._structured(self.distribution.mean)

    @property
    def logvar(self) -> torch.Tensor:
        return self._structured(self.distribution.logvar)

    def sample(self) -> torch.Tensor:
        return self._structured(self.distribution.sample())

    def mode(self) -> torch.Tensor:
        return self._structured(self.distribution.mode())

    def kl(self) -> torch.Tensor:
        return rearrange(
            self.distribution.kl(),
            "(b v) -> b v",
            b=self.batch_size,
            v=self.views,
        )
