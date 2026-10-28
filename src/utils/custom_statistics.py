import torch


def get_model_params(model: torch.nn.Module):
    """Get total number of params in model."""
    total_params = sum(torch.numel(p) for p in model.parameters())
    net_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum(torch.numel(p) for p in net_parameters)
    return {'Model': model.__class__,
            'Total params': total_params,
            'Trainable params': trainable_params,
            'Non-trainable params': total_params - trainable_params}
