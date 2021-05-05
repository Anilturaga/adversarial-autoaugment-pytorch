""" Driver script for training with Adversarial AutoAugment.
        [1] Training with train_augment
        [2] Results from test dataset
"""
__author__ = "Anil Turaga"
__copyright__ = "Untangle License"
__version__ = "2.0.0"
__status__ = "Development"
__date__ = "03/May/2021"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms
import pickle
from untangle import UntangleAI

class TrainAugmentArgs:
    """
    A new directory 'mname' is created with sub-directory named train_augment/'experiment_ID' 
    """
    mname = 'test_net'
    num_class = 10
    batch_size = 128
    epoch = 600
    gpu_count = 8
    experiment_ID = 1

class LeNet(nn.Module):
    # TODO: This isn't really a LeNet, but we implement this to be
    #  consistent with the Evidential Deep Learning paper
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = None
        lenet_conv = []
        lenet_conv += [nn.Conv2d(3, 6, 5)] 
        lenet_conv += [torch.nn.ReLU(inplace=True)]
        lenet_conv += [nn.MaxPool2d(2, 2)]
        lenet_conv += [nn.Conv2d(6, 16, 5)]
        lenet_conv += [torch.nn.ReLU(inplace=True)]
        lenet_conv += [nn.MaxPool2d(2, 2)]
        lenet_dense = []
        lenet_dense += [nn.Linear(16 * 5 * 5, 120)]
        lenet_dense += [torch.nn.ReLU(inplace=True)]
        lenet_dense += [nn.Linear(120, 84)]
        lenet_dense += [torch.nn.ReLU(inplace=True)]
        lenet_dense += [nn.Linear(84, 10)]

        self.features = torch.nn.Sequential(*lenet_conv)
        self.classifier = torch.nn.Sequential(*lenet_dense)

    def forward(self, input):
        output = self.features(input)
        output = output.view(-1, 16 * 5 * 5)
        output = self.classifier(output)
        return output

if __name__ == '__main__':
    args= TrainAugmentArgs()
    untangleai = UntangleAI()
    model = LeNet()
    #No need to use transforms.ToTensor as train_augment handles it internally
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
    #Empty transforms.compose([]) should be used if there are no transforms. e.g. MNIST dataset
    transform_test = transforms.Compose([])
    trainset = torchvision.datasets.CIFAR10(root='./testDataset/', train=True, download=True,transform=transform_train,)
    testset = torchvision.datasets.CIFAR10(root='./testDataset/', train=False, download=True,transform=transform_test)
    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9, nesterov = True, weight_decay = 1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max = 2)
    
    #train_augment returns the trained model(best loss model)
    trained_model = untangleai.train_augment(model,optimizer,scheduler,trainset,testset,args)
    
    #Checkpoint models such as model at the end of each epoch are saved in 'mname'/train_augment/'experiment_ID'/models/ for custom experiments
    exp_path = untangleai.train_augment_path # test_net/train_augment/1/
    
    #Logs such as policies and losses are saved in 'mname'/train_augment/'experiment_ID'/logs.pkl
    with open('test_net/train_augment/1/logs.pkl', 'rb') as f: 
        data = pickle.load(f)
        print(data) 


