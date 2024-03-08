# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, weight):
        loss = F.binary_cross_entropy_with_logits(
            input=output, target=target, weight=weight
        )
        return loss
