import math
import torch
import torch.nn as nn
from model.components.LPN_component import ClassBlock
from utils.commons import print_nb_params
from collections import OrderedDict
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


class LPN_Module(nn.Module):
    def __init__(self,num_classes,pool='avg',num_blocks=4, feature_dim=2048, return_f=False):
        super(LPN_Module, self).__init__()
        self.num_classes = num_classes
        self.pool = pool
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.return_f = return_f
        if self.pool == 'avg+max':
            in_feature_dim = 2 * feature_dim
        else:
            in_feature_dim = feature_dim

        for i in range(self.num_blocks):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(in_feature_dim, num_classes, 0.5, return_f=return_f))

    def forward(self,x):
        #input:[B,C,H,W]
        if len(x.shape)<4:
            if x.shape[1]!=self.feature_dim:
                x = x.permute(0,2,1)
            x = x.reshape(x.shape[0], x.shape[1], int(math.sqrt(x.shape[2])), -1)
        if x.shape[-1]==self.feature_dim:
            x = x.permute(0,3,1,2)

        pooling_methods = {
            'avg': lambda x: self.get_part_pool(x, pool='avg'),
            'max': lambda x: self.get_part_pool(x, pool='max')
        }
        if self.pool == 'avg+max':
            x = torch.cat((pooling_methods['avg'](x), pooling_methods['max'](x)), dim=1)
        else:
            x = pooling_methods[self.pool](x)

        x = x.view(x.size(0), x.size(1), -1)
        y = self.part_classifier(self.num_blocks,x,cls_name='classifier')

        '''
        since LPN only uses the class digit during the training process, 
        we return "None" feature value to the framework
        '''
        # if self.training:
        #     return y, None

        #to align with QDFL method
        if self.training:
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
            else:
                return y, None
        return y
        # return x

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.num_blocks),W/(2*self.num_blocks)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.num_blocks-c_h)*2, W+(self.num_blocks-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear')
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.num_blocks),W/(2*self.num_blocks)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.num_blocks):
            i = i + 1
            if i < self.num_blocks:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.insert(0, avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.insert(0, avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, num_blocks, x, cls_name='classifier'):
        # [B, dim, C]
        part = {}
        predict = {}
        for i in range(num_blocks):
            # [B, dim, 1]-->[B, dim]
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            # 调用transformer中的特定模块进行预测，如classifier_heat1
            name = cls_name + str(i)
            c = getattr(self, name)
            # 如classifier_heat{i}([B, dim, 1](i))
            predict[i] = c(part[i])
        y = []
        # y = [num_blocks,[B,C_Num]]
        for i in range(num_blocks):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

def main():
    x = torch.randn(20, 768, 30, 30)
    m = LPN_Module(num_classes=701, pool='avg+max', num_blocks=4, feature_dim=768, return_f=True).train()
    z,_ = m(x)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    # if isinstance(z,tuple):
    #     print(f'Training Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    # else:
    #     print(f'Training Output shape is {torch.stack(z,dim=0).shape}')

    # m.train(False)
    z = m(x)
    if isinstance(z,tuple):
        print(f'Evaluating Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    else:
        print(f'Evaluating Output shape is {z.shape}')

if __name__ == "__main__":
    main()