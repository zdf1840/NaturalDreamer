import torch
import torch.nn.functional as F
import utils


class TwoHotCategoricalStraightThrough(torch.distributions.Distribution):
    def __init__(self, logits: torch.Tensor, bins: int = 255, low: float = -20.0, high: float = 20.0):
        super().__init__(validate_args=False)
        self.logits = logits
        self.bin_centers = torch.linspace(low, high, bins, device=logits.device)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = utils.symlog(value).clamp(self.bin_centers[0], self.bin_centers[-1])
        indices = ((value - self.bin_centers[0]) / (self.bin_centers[1] - self.bin_centers[0])).clamp(0, len(self.bin_centers) - 1)

        lower = indices.floor().long().unsqueeze(-1)
        upper = indices.ceil().long().unsqueeze(-1)
        alpha = (indices - lower.squeeze(-1)).unsqueeze(-1)

        probs = F.softmax(self.logits, dim=-1)
        return torch.log((1 - alpha) * probs.gather(-1, lower) + alpha * probs.gather(-1, upper)).squeeze(-1)

    @property
    def mean(self) -> torch.Tensor:
        return utils.symexp((F.softmax(self.logits, dim=-1) * self.bin_centers).sum(-1, keepdim=True))

