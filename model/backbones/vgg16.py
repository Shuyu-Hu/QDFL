import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
from utils.commons import print_nb_params


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    def __init__(self, n_classes=None):
        super(VGG16, self).__init__()
        self.n_classes = n_classes

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        if self.n_classes:

            # FC layers
            self.fc1 = vgg_fc_layer(7*7*512, 4096)
            self.fc2 = vgg_fc_layer(4096, 4096)

            # Final layer
            self.cls = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        if self.n_classes:
            out = vgg16_features.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.cls(out)
            return out
        else:
            return vgg16_features

def main():
    x = torch.randn(1, 3, 256, 256)
    m = VGG16(n_classes=None)
    r = m(x)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')
    print(list(m.parameters())[-1].shape[0])


if __name__ == '__main__':
    main()