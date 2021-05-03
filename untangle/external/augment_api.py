"""
File should reside in external folder, all functions should reside in adversarial_auto_augment
folder alongside external folder

FUNCTIONS:
    main function to take model, dataloaders
    Evaluate takes model, dataloader for test and corrupstion datasets
    Logs dir and model checkpoints automatically created complying with version2 

"""
import os
import torch
import torch.multiprocessing as mp
from untangle.adversarial_auto_augment.main import main

def train_augment(model,optimizer,scheduler,train_dataset,valid_dataset,args,exp_path):
    '''
    Trains network using Adversarial AutoAugment
    :param model: pytorch model object
    :param optimizer: model's optimizer. e.g. SGD 
    :param scheduler: model's scheduler. e.g. CosineAnnealingLR
    :param train_dataset: Pytorch dataset object
    :param valid_dataset: Pytorch dataset object
    :param scheduler: model's scheduler. e.g. CosineAnnealingLR
    :param args: data specific params (ref to TrainAugmentArgs class)
    :return: Trained model object
    '''
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '27272'
    except:
        print("Please kill process at port 27272")
    mp.spawn(main,
             args=(model,optimizer,scheduler,train_dataset,valid_dataset,args,exp_path),
             nprocs=args.gpu_count,
             join=True)
    ckpt = os.path.join(os.path.join(exp_path,'models/best_top1.pth'))
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    return model