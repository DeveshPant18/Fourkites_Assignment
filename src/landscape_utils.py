import torch
import copy

def get_params(model):
    """Returns a list of parameter tensors from the model."""
    return [p.data.clone() for p in model.parameters()]

def get_random_dir(model):
    """
    Generates a random direction vector (delta) with Filter Normalization.
    """
    direction = []
    for param in model.parameters():
        # 1. Generate random Gaussian noise
        d = torch.randn_like(param)
        
        # 2. Normalize filter-wise (for conv layers) or layer-wise
        if param.dim() > 1: # Weights (skip bias)
            # Calculate norms (flatten all dims except the first)
            d_norm = d.view(d.size(0), -1).norm(dim=1, keepdim=True)
            p_norm = param.view(param.size(0), -1).norm(dim=1, keepdim=True)
            
            scale = p_norm / (d_norm + 1e-10)
            
            # Create a shape like [32, 1, 1, 1] for broadcasting
            view_shape = [d.size(0)] + [1] * (d.dim() - 1)
            d = d * scale.view(view_shape)
            
        direction.append(d)
    return direction

def set_weights(model, origin, direction_1, direction_2, alpha, beta):
    """
    Sets model weights to: theta* + alpha * dir1 + beta * dir2
    """
    if direction_2 is None:
        # 1D Case: Only iterate over origin and direction_1
        for param, o, d1 in zip(model.parameters(), origin, direction_1):
            param.data = o + alpha * d1
    else:
        # 2D Case: Iterate over origin, direction_1, and direction_2
        for param, o, d1, d2 in zip(model.parameters(), origin, direction_1, direction_2):
            param.data = o + alpha * d1 + beta * d2