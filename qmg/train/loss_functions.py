
import torch
import torch.nn as nn

class valid_state_loss(nn.Module):
    def __init__(self, valid_state_mask: torch.tensor, reduction="mean"):
        """
        Parameters
        ----------
        valid_state_mask :  torch.tensor
            binart tensor, 1 indicates valid quantum state, and 0 indicates invalid.
        """
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.reduction = reduction

    def forward(self, predictions):
        loss = (predictions * self.valid_state_mask).sum(dim=1)
        if self.reduction == "mean":
            return torch.mean(-torch.log(loss))
        elif self.reduction == "sum":
            return torch.sum(-torch.log(loss))
        else:
            return -torch.log(loss)
        
class jenson_shannon_divergence(nn.Module):
    def __init__(self, valid_state_mask, reduction="batchmean"):
        """
        Parameters
        ----------
        valid_state_mask :  torch.tensor
            binart tensor, 1 indicates valid quantum state, and 0 indicates invalid.
        """
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)
        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, outputs, coverted_noise):
        outputs = outputs[:, self.valid_state_mask == 1.]
        coverted_noise = coverted_noise[:, self.valid_state_mask == 1.]
        converted_noise_probability = self.softmax_layer(coverted_noise)
        total_m = 0.5 * (outputs + converted_noise_probability)
        loss = 0.0
        loss += self.kl_div(outputs.log(), total_m) 
        loss += self.kl_div(converted_noise_probability.log(), total_m) 
        return (0.5 * loss)
    
class MMDLoss(nn.Module):
    """Useless loss function."""
    def __init__(self, valid_state_mask, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma
        self.valid_state_mask = valid_state_mask

    def gaussian_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.unsqueeze(1).expand(x_size, y_size, dim)
        y = y.unsqueeze(0).expand(x_size, y_size, dim)

        kernel_input = (x - y).pow(2).mean(2) / (2 * self.sigma)
        return torch.exp(-kernel_input)

    def forward(self, x, y):
        xx = self.gaussian_kernel(x, x)
        yy = self.gaussian_kernel(y, y)
        xy = self.gaussian_kernel(x, y)
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
class diversity_loss(nn.Module):
    def __init__(self, valid_state_mask, reduction="batchmean"):
        super().__init__()
        self.valid_state_mask = valid_state_mask
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=False)

    def jensen_shannon_divergence(self, ps: torch.tensor, qs: torch.tensor):
        m = 0.5 * (ps + qs)
        return 0.5 * (self.kl_div(ps.log(), m) + self.kl_div(qs.log(), m))
    
    def forward(self, distributions):
        distributions = distributions[:, self.valid_state_mask == 1.]
        reversed_distributions = torch.flip(distributions, dims=[0])
        return - self.jensen_shannon_divergence(distributions, reversed_distributions)