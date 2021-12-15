import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg16,resnet18

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()        
        # pretrained VGG16 model from torchvision
        self.vgg = vgg16(pretrained=True)        
        # all conv_layers
        self.conv_layers = self.vgg.features[:30]        
        # max_pool and avgpool layer after last conv layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)    
        self.avgpool=self.vgg.avgpool    
        # the classifier of vgg16
        self.classifier = self.vgg.classifier
        # saved variable for gradient of the last conv_layer
        self.gradients = None    
    # hook function to obtain the gradient
    def features_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # get features maps
        x = self.conv_layers(x)        
        # register the hook
        h = x.register_hook(self.features_hook)        
        # continue to forward
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x    
    def get_gradient(self):
        return self.gradients    
    def get_features_map(self, x):
        return self.conv_layers(x)

class Resnet_18(nn.Module):
    def __init__(self):
        super(Resnet_18,self).__init__()
        self.resnet_18=resnet18(pretrained=True)
        self.conv_layers=nn.Sequential(
            self.resnet_18.conv1,
            self.resnet_18.bn1,
            self.resnet_18.relu,
            self.resnet_18.maxpool,
            self.resnet_18.layer1,
            self.resnet_18.layer2,
            self.resnet_18.layer3,
            self.resnet_18.layer4,
        )
        self.avgpool=self.resnet_18.avgpool
        self.fc=self.resnet_18.fc
        self.gradients = None
    
    # hook function to obtain the gradient
    def features_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.conv_layers(x)
        h = x.register_hook(self.features_hook)
        # continue to forward
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.fc(x)
        return x    
    def get_gradient(self):
        return self.gradients
    def get_features_map(self, x):
        return self.conv_layers(x)