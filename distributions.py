import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ) -> None:
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x).clamp(self.low, self.high)
        x_expanded = x.unsqueeze(-1)
        below = (self.bins <= x_expanded).sum(dim=-1, keepdim=True) - 1
        above = below + 1
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        below = torch.maximum(below, torch.zeros_like(below))
        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x_expanded))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x_expanded))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below.squeeze(-1).long(), len(self.bins)) * weight_below
            + F.one_hot(above.squeeze(-1).long(), len(self.bins)) * weight_above
        )
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(dim=self.dims)
