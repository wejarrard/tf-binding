# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assign class weights for each element in the batch
        class_weights = targets * self.weight[1] + (1 - targets) * self.weight[0]

        # Calculate Binary Cross Entropy loss with class weights
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", weight=class_weights
        )

        # Compute the probability
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0

        # Apply the focal loss formula
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        # Aggregate the loss based on reduction method
        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        return F_loss



# Example of usage:
# criterion = FocalLoss(alpha=1, gamma=2)
