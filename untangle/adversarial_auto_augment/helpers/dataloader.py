import os

import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist

from torch.utils.data import Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
from untangle.adversarial_auto_augment.helpers.transform import train_collate_fn, test_collate_fn

def get_dataloader(trainset, testset, args, multinode = False):
    batch_size = args.batch_size // dist.get_world_size()
    train_sampler = None
    test_sampler = None
    if multinode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        test_sampler = torch.utils.data.distributed.DistributedSampler(
                testset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True if train_sampler is None else False, num_workers=0, pin_memory=True,
        sampler=train_sampler, drop_last=True, collate_fn = train_collate_fn)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
        sampler=test_sampler, drop_last=False, collate_fn = test_collate_fn
    )
    
    return train_sampler, train_loader, test_loader