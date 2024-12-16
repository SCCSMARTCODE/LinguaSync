import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


def cross_entropy_loss(ignore_index):
    """
    Creates a CrossEntropyLoss instance with optional padding token ignored.
    Args:
        ignore_index (int): Index of the token to ignore in the loss (e.g., [PAD]).
    Returns:
        nn.CrossEntropyLoss: Loss function instance.
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def get_optimizer(model, learning_rate, weight_decay):
    """
    Configures the Adam optimizer with weight decay.
    Args:
        model (nn.Module): The model whose parameters will be optimized.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization.
    Returns:
        optim.Adam: Configured optimizer.
    """
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_scheduler(optimizer, gamma):
    """
    Configures an Exponential Learning Rate Scheduler.
    Args:
        optimizer (optim.Optimizer): The optimizer to apply the scheduler to.
        gamma (float): Decay factor for the learning rate.
    Returns:
        ExponentialLR: Learning rate scheduler.
    """
    return ExponentialLR(optimizer, gamma=gamma)


def gradient_clipping(model, max_norm):
    """
    Clips gradients to a maximum norm.
    Args:
        model (nn.Module): The model whose gradients will be clipped.
        max_norm (float): Maximum norm for gradient clipping.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
