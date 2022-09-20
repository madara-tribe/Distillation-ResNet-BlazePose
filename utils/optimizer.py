import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def create_optimizer(model, config):
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, betas=(0.9, 0.999), eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(params=[
        {'params': model.parameters(), 'lr': config.lr},
        ], lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.t_max, eta_min=config.eta_min)
    return optimizer, scheduler
