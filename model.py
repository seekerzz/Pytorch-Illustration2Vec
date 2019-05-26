import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision.models import vgg16
from torchsummary import summary
import sklearn


class I2V(nn.Module):
    def __init__(self,dropout_rate=0.3):
        super().__init__()
        vgg_model = vgg16(pretrained=True)
        self.feature_extractor = vgg_model.features
        top_layers = [nn.Conv2d(512,1024,3,1,1),
                      nn.Conv2d(1024,1024,3,1,0),
                      nn.Conv2d(1024,512,3,1,0),
                      nn.AvgPool2d(4),
                      nn.Dropout(dropout_rate),
                      nn.Sigmoid(),
                      ]
        vgg_model = vgg16(pretrained=True)
        self.feature_extractor = vgg_model.features
        top_conv_layers = [
            nn.Conv2d(512,1024,4,2,1),
            nn.Conv2d(1024,1024,4,2,1),
        ]
        top_dense_layers =[
            nn.Linear(1024*2*2,2048),
            nn.Linear(2048,512),
            nn.Dropout(dropout_rate),
            nn.Sigmoid()
        ]
        #self.top_conv_layers = nn.Sequential(*top_conv_layers)
        #self.top_dense_layers = nn.Sequential(*top_dense_layers)
        self.top_layers = nn.Sequential(*top_layers)

    def forward(self, img):
        f = self.feature_extractor(img)
        out = self.top_layers(f)
        out = out.view(out.shape[0],-1)
        '''
        out = self.top_conv_layers(f)
        out = out.view(out.size(0),-1)
        out = self.top_dense_layers(out)
        '''
        return out




if __name__=="__main__":
    i2v=I2V().cuda()
    summary(i2v,(3,256,256))
