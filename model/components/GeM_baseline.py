import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x

class GeM_Baseline(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """

    def __init__(self, in_channels, cls_token, num_classes, return_f, p=3, eps=1e-6):
        super().__init__()
        self.return_f = return_f
        self.cls_token = cls_token
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.gem_feature_classifier = ClassBlock(in_channels, num_classes, 0.5, return_f=return_f)
        if self.cls_token:
            self.token_classifier = ClassBlock(in_channels, num_classes, 0.5, return_f=return_f)
    def forward(self, x):
        if self.cls_token:
            t, x = x
            token_feature = self.token_classifier(t)

        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
        x = x.flatten(1)
        x = F.normalize(x, p=2, dim=1)
        y = self.gem_feature_classifier(x)
        if self.cls_token:
            if self.training:
                y = [y] + [token_feature]
                if self.return_f:  # Use contrast learning or not
                    cls, features = [], []
                    for i in y:
                        cls.append(i[0])
                        features.append(i[1])
                    return cls, features
                else:
                    return y, None
            else:
                token_feature = token_feature.view(token_feature.size(0), -1, 1)
                y = torch.cat([y.unsqueeze(-1), token_feature], dim=2)
                # y = torch.cat([y,token_feature], dim=2)
        else:
            if self.training:
                y = [y]
                if self.return_f:
                    cls, features = [], []
                    for i in y:
                        cls.append(i[0])
                        features.append(i[1])
                    return cls, features
                else:
                    return y, None
            else:
                y = y
        return y


def main():
    h = 32
    # hh = h ** 2
    ################
    in_dim = 768
    batch_size = 24

    # x = torch.randn(batch_size, h, h, in_dim)#b,c,h,w
    # x = [torch.randn(5, in_dim), torch.randn(5, in_dim, 12, 12)]
    x = torch.randn(batch_size, in_dim, h, h)  # b,c,h,w
    agg = GeM_Baseline(in_channels=768,cls_token=False,num_classes=701,return_f=True).eval()
    z = agg(x)
    print(z.shape)


if __name__ == '__main__':
    main()