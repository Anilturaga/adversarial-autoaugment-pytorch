"""core module to train a model with Adversarial AutoAugment
        [1] Handles distributed data parallel
        [2] Handles model and controller training
        [3] Handles logging and model checkpoints for each epoch
"""

# Copyright (c) 2021. UntangleAI PTE LTD. All rights reserved.
# Proprietary and confidential. Copying and distribution is strictly prohibited.

__author__ = "Anil Turaga"
__copyright__ = "Untangle AI pvt ltd"
__version__ = "2.0.0"
__maintainer__ = "Anil Turaga"
__email__ = "turagaanil@gmail.com"
__status__ = "Development"
__date__ = "01/May/2021"

import numpy as np
import torch

import sys
from tqdm import tqdm
from untangle.adversarial_auto_augment.helpers.dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
from untangle.adversarial_auto_augment.helpers.transform import parse_policies, MultiAugmentation
from untangle.adversarial_auto_augment.helpers.utils import *
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from untangle.adversarial_auto_augment.helpers.transform import test_collate_fn
from torchvision.transforms import transforms

def init_ddp(local_rank,gpu_count):
    if local_rank !=-1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method='env://',rank = local_rank,world_size = gpu_count)
 
def main(local_rank,model,optimizer,scheduler,train_dataset,valid_dataset,args,exp_path):
    seed_everything()
    logger = Logger(exp_path)
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    init_ddp(local_rank,args.gpu_count)
    print("EXPERIMENT:",args.mname)
    print()
    train_sampler, train_loader, test_loader = get_dataloader(train_dataset,valid_dataset, args, multinode = (local_rank!=-1))    
    controller = get_controller(local_rank)
    device = torch.device('cuda', local_rank)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters = True)
    controller_optimizer = Adam(controller.parameters(), lr = 0.00035)
    criterion = CrossEntropyLabelSmooth(num_classes = args.num_class)
    scaler = GradScaler()
    step = 0
    M = 8
    for epoch in range(args.epoch):
        if local_rank >=0:
            train_sampler.set_epoch(epoch)
        
        Lm = torch.zeros(M).cuda()
        Lm.requires_grad = False

        
        model.train()
        controller.train()
        policies, log_probs, entropies = controller(M) # (M,2*2*5) (M,) (M,) 
        policies = policies.cpu().detach().numpy()
        parsed_policies = parse_policies(policies)
        if epoch == 0:
            train_loader.dataset.transform.transforms.append(MultiAugmentation(parsed_policies))
            train_loader.dataset.transform.transforms.append(transforms.Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])))
            test_loader.dataset.transform.transforms.append(transforms.ToTensor())
            
        else:
            train_loader.dataset.transform.transforms[-2] = MultiAugmentation(parsed_policies)
        train_loss = 0
        train_top1 = 0
        train_top5 = 0
        
        progress_bar = tqdm(train_loader)
        for idx, (data,label) in enumerate(progress_bar):
            optimizer.zero_grad()
            data = data.cuda()
            label = label.cuda()
            with autocast(enabled=True):
                pred = model(data)
                losses = [criterion(pred[i::M,...] ,label) for i in range(M)]
                loss = torch.mean(torch.stack(losses))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()        

            for i,_loss in enumerate(losses):
                Lm[i] += reduced_metric(_loss.detach(), args.gpu_count, local_rank !=-1) / len(train_loader)
            
            top1 = None
            top5 = None
            for i in range(M):
                _top1,_top5 = accuracy(pred[i::M,...], label, (1, 5))
                top1 = top1 + _top1/M if top1 is not None else _top1/M
                top5 = top5 + _top5/M if top5 is not None else _top5/M
            
            train_loss += reduced_metric(loss.detach(), args.gpu_count, local_rank !=-1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), args.gpu_count, local_rank !=-1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), args.gpu_count, local_rank !=-1) / len(train_loader)
            
            progress_bar.set_description('Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, args.epoch, idx + 1, len(train_loader), loss.item()))
            step += 1
        
        model.eval()
        controller.train()
        controller_optimizer.zero_grad()

        normalized_Lm = (Lm - torch.mean(Lm))/(torch.std(Lm) + 1e-5)
        score_loss = torch.mean(-log_probs * normalized_Lm) # - derivative of Score function
        def_entropy_penalty = 1e-5 #FIX - Make it as args in future release
        entropy_penalty = torch.mean(entropies) # Entropy penalty
        controller_loss = score_loss - def_entropy_penalty * entropy_penalty

        controller_loss.backward()
        controller_optimizer.step()
        scheduler.step()

        valid_loss = 0.
        valid_top1 = 0.
        valid_top5 = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (data,label) in enumerate(tqdm(test_loader)):
                b = data.size(0)
                data = data.cuda()
                label = label.cuda()
                
                pred = model(data)
                loss = criterion(pred,label)

                top1, top5 = accuracy(pred, label, (1, 5))
                valid_loss += reduced_metric(loss.detach(), args.gpu_count, local_rank !=-1) *b 
                valid_top1 += reduced_metric(top1.detach(), args.gpu_count, local_rank !=-1) *b
                valid_top5 += reduced_metric(top5.detach(), args.gpu_count, local_rank !=-1) *b 
                cnt += b
            
            valid_loss = valid_loss / cnt
            valid_top1 = valid_top1 / cnt
            valid_top5 = valid_top5 / cnt
        
        logger.add_dict(
            {
                'train_loss' : train_loss,
                'train_top1' : train_top1,
                'train_top5' : train_top5,
                'controller_loss' : controller_loss.item(),
                'score_loss' : score_loss.item(),
                'entropy_penalty' : entropy_penalty.item(),
                'valid_loss' : valid_loss,
                'valid_top1' : valid_top1,
                'valid_top5' : valid_top5,
                'policies' : parsed_policies,
            }
        )
        if local_rank <= 0:
            logger.save_model(model,epoch)
        logger.info(epoch,['train_loss','train_top1','train_top5','valid_loss','valid_top1','valid_top5','controller_loss'])
    
        logger.save_logs()
    return model
