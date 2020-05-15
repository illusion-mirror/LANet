import  torch
import torch.nn as nn
import torch.nn.functional
from torch.nn import init
from collections import OrderedDict
class LinearBottleneck(nn.Module):
    def __init__(self,inplanes,outplanes,stride,t,activation=nn.ReLU6):
        '''
        :param inplanes:
        :param outplanes:
        :param stride:
        :param t:
        :param activation:
        '''
        '''
        首先利用点卷积升维，然后利用深度卷积计算，最后利用点卷积降维，每个卷积后跟着BN和激活函数
        '''
        super(LinearBottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,t*inplanes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*t)
        self.conv2 = nn.Conv2d(inplanes*t,inplanes*t,kernel_size=3,stride=stride,bias=False,groups=t*inplanes)
        self.bn2 = nn.BatchNorm2d(inplanes*t)
        self.conv3 = nn.Conv2d(inplanes*6,outplanes,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.activation = activation(inplace=True)
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.bn3(out)
            #out = self.activation(out)
            if stride==1 and self.inplanes==self.outplanes:
                out = out+residual
            return out

    class SPP(nn.Module):
        def __init__(self,in_channels):
            nn.Module.__init__()
            self.pool1 = x1 = nn.AvgPool2d([32,32])
            self.pool2 = nn.AvgPool2d([8,8])
            self.pool3 = nn.AvgPool2d([2,2])
            self.conv1 = nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1, bias=False,
                                   groups=in_channels)
            self.conv2 = nn.Conv2d(in_channels,in_channels//3,kernel_size=1,bias=False)
            self.conv3 = nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1, bias=False,
                      groups=in_channels)
            self.conv4 = nn.Conv2d(in_channels, in_channels // 3, kernel_size=1, bias=False)
            self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False,
                      groups=in_channels)
            self.conv6 = nn.Conv2d(in_channels, in_channels // 3, kernel_size=1, bias=False)
            #torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)

            def forward(self, x):
                out = x
                x1 = self.pool1(x)
                x2 = self.pool2(x)
                x3 = self.pool3(x)
                x1 = self.conv1(x1)
                x1 = self.conv2(x1)
                x2 = self.conv3(x2)
                x2 = self.conv4(x2)
                x3 = self.conv5(x3)
                x3 = self.conv6(x3)
                x1 = nn.functional.interpolate(x1, size=[], mode='bilinear')
                x2 = nn.functional.interpolate(x2, size=[], mode='bilinear')
                x3 = nn.functional.interpolate(x3, size=[], mode='bilinear')
                x4 = torch.cat([x1,x2,x3,out],2)
                return x4

    base = {'352': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

    def vgg16(cfg, i, batch_norm=False):
        layer = []
        inchannels = i
        for v in cfg:
            if v == 'M':
                layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = LinearBottleneck()
                if batch_norm:
                    layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layer += [conv2d, nn.ReLU(inplace=True)]
                inchannels = v
        return layer

    class LANet(nn.Module):
        def __init__(self,vgg,spp):
            nn.Module.__init__()
            self.vgg = nn.ModuleList(vgg)
            self.spp = spp
        def forward(self,x):
            for i in range():
                x = self.vgg[i](x)
            x = self.spp(x)
            x = nn.functional.interpolate(x,scale_factor=16)
            return x




