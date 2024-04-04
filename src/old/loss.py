# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:
            return F_loss


# Example of usage:
# criterion = FocalLoss(alpha=1, gamma=2)
