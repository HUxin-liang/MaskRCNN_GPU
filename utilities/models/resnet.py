import torch
import torch.nn as nn
import math
from utilities.models.bottleneck import Bottleneck
from utilities.models.samePad2d import SamePad2d

class ResNet(nn.Module):
    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.inplanes = 64
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2)
        )
        # self.layers = [3,4,6,3] or [3,4,23,3]
        self.C2 = self.make_layers(self.block, 64, self.layers[0])
        self.C3 = self.make_layers(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layers(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layers(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!= 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01)
            )
        layers = []
        # 该部分是将每个convx_x(即blocks)的第一个residual子结构保存在layers列表中
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 该部分是将每个convx_x的剩下residual子结构保存在layers列表中，这样就完成了一个blocks的构造
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)



