""" Driver script for signal estimation services.
        [1] signal estimation
        [2] signal attribution of a point
"""
__author__ = "Rahul Soni"
__copyright__ = "Untangle License"
__version__ = "1.0.6"


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms

from untangle import UntangleAI
class TrainAugmentArgs:
    num_gpus = 8
    num_class = 10
    batch_size = 128
    mname = 'test_net'
    epoch = 600
    experiment_ID = 7
    gpu_count = 8

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
    #No need to use transforms.ToTensor as we do it internally
    #Empty transforms.compose([]) can also be used
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
    transform_test = transforms.Compose([])
    trainset = torchvision.datasets.CIFAR10(root='./testDataset/', train=True, download=True,transform=transform_train,)
    testset = torchvision.datasets.CIFAR10(root='./testDataset/', train=False, download=True,transform=transform_test)
    optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9, nesterov = True, weight_decay = 1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max = 2)
    trained_model = untangleai.train_augment(model,optimizer,scheduler,trainset,testset,args,1)