'''
description: 
param : 
return: 
Author: xinyebei@xinye.com
Date: 2025-04-23 14:14:49
LastEditors: xinyebei@xinye.com
'''
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)

        return loss