import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einops
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
from utils.commons import print_nb_params

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False,KD=False):
        super(ClassBlock, self).__init__()
        self.KD = KD
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
        if self.training or self.KD:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x

class MLP(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """

    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=1):
        super(MLP, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'):  # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, x_out21

class DSA(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_mlp=1):
        super(DSA, self).__init__()
        self.proj = MLP(in_channels=in_channels, hid_channels=hid_channels, out_channels=out_channels, norm_layer=None,
                        num_mlp=num_mlp)

    def forward(self, x):
        # [B,C,H,W]->[B,C+256,H*W]
        x = x.flatten(2)
        w = self.proj(x)
        w = F.softmax(F.normalize(w, dim=1), dim=2)

        feat = torch.cat([x, w], dim=1)

        return feat

class DAC(nn.Module):
    def __init__(self,num_classes,block=4,return_f=False,in_channels=1024,KD=False):
        super(DAC,self).__init__()
        self.KD=KD
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.return_f = return_f
        self.DSA = DSA(self.in_channels, 2*self.in_channels, 256)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_channels, num_classes, 0.5, return_f=return_f,KD=self.KD)
        self.block = block
        self.tri_layer = TripletAttention()
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_channels, num_classes, 0.5, return_f=self.return_f,KD=self.KD))

    def forward(self,x):
        t, x = x
        if self.training or self.KD:
            # 1. Domain Space Alignment Module
            b, c, h, w = x.shape
            pfeat = x.flatten(2)  # (bs, c, h*w)
            pfeat_align = self.DSA(pfeat)
            # pfeat_align = pfeat

            # 2. triplet attention
            tri_features = self.tri_layer(x)  # 旋转另外两个轴，返回两个一样的特征体(bs, 1024, 12, 12); (bs, 1024, 12, 12)
            convnext_feature = self.classifier1(t)  # class: (bs, 701); feature: (bs, 512)
            tri_list = []
            for i in range(self.block):
                tri_list.append(tri_features[i].mean([-2, -1]))  # average pooling, 一张图变一个像素
            triatten_features = torch.stack(tri_list, dim=2)  # 把另外两个轴旋转的特征体
            if self.block == 0:
                y = []
            else:
                y = self.part_classifier(self.block, triatten_features,
                                         cls_name='classifier_mcb')  # 把另外两个轴旋转的feature也做分类
            y = y + [convnext_feature]  # 三个分支连起来
            if self.return_f:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return pfeat_align, cls, features, t, x
        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass
        return t
    
    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training and not self.KD:
            return torch.stack(y, dim=2)
        return y

def main():
    out_dim = 1024
    x = [torch.randn(2, out_dim), torch.randn(2, out_dim, 12, 12)]
    agg =  DAC(num_classes=701,block=2,return_f=0.3).train()
    # print(agg)
    print_nb_params(agg,'classifier')
    x_out = agg(x)
    # evaluate_model(agg, x, 0)
    print(x_out.shape)
    # print(output.shape)
    # print(output.dtype)



if __name__ == '__main__':
    main()