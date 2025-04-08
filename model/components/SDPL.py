import math
import torch
import torch.nn as nn
from torch.nn import init
# from model.components.LPN_component import ClassBlock
from utils.commons import print_nb_params,evaluate_model
from collections import OrderedDict
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

def num_sdpl_block(num_blocks):
    if num_blocks==1:
        return 1
    return int(num_blocks*(1+num_blocks)/2)

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

class SDPL_Module(nn.Module):
    def __init__(self, num_classes, pool='avg', num_blocks=4, feature_dim=2048, stride=1):
        super(SDPL_Module, self).__init__()
        self.num_classes = num_classes
        self.pool = pool
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.stride = stride

        self.bn = nn.BatchNorm1d(feature_dim, affine=False)  # 1024->2048

        # 常规的初始化 Gem 或 Gem_new1
        for i in range(10):
            gem = 'gem' + str(i)
            setattr(self, gem, GeM(feature_dim))

        # Create shift-guided parameters， 偏移分区软融合的时候需要初始化网络参数。 使用 get_part_pool_shift_new3 的时候需要解除注释
        self.shift_blocks = nn.Sequential(torch.nn.Conv2d(feature_dim, 1024, kernel_size=1),
                                          # stage4的channels, 可以尝试换成maxpooling和avgpooling处理通道维度
                                          torch.nn.ReLU(),
                                          torch.nn.AdaptiveAvgPool2d((1, 1)),  # 1 1024 1 1
                                          torch.nn.Flatten(),
                                          torch.nn.Linear(1024, 512),  # H * W！！
                                          torch.nn.Dropout(0.5),
                                          torch.nn.Linear(512, 512),
                                          torch.nn.Dropout(0.5),
                                          torch.nn.Linear(512, 3),  # 三块进行softmax
                                          torch.nn.Softmax(dim=1)
                                          )

        for i in range(num_sdpl_block(self.num_blocks)):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(feature_dim, self.num_classes, 0.5))

    def forward(self,x):
        #input:[B,C,H,W]
        if len(x.shape)<4:
            if x.shape[1]!=self.feature_dim:
                x = x.permute(0,2,1)
            x = x.reshape(x.shape[0], x.shape[1], int(math.sqrt(x.shape[2])), -1)
        if x.shape[-1]==self.feature_dim:
            x = x.permute(0,3,1,2)
        #[B,C,H,W]
        if self.pool == 'avg+max':
            x1 = self.get_part_pool_dense(x, pool='avg')
            x2 = self.get_part_pool_dense(x, pool='max')
            x = torch.cat((x1, x2), dim=1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool_dense_shift_new3(x, pool1_gem=True)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'max':
            x = self.get_part_pool_dense(x)
            x = x.view(x.size(0), x.size(1), -1)

        # x = x.view(x.size(0), x.size(1), -1)
        y = self.part_classifier(x,cls_name='classifier')

        '''
        since LPN only uses the class digit during the training process, 
        we return "None" feature value to the framework
        '''
        if self.training:
            return y, None
        return y
        # return x

    def get_part_pool_dense(self, x, block=4, pool='avg', pool1_gem=False):
        result = []
        base_feature = []
        pixels_n = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)  # 计算input的宽高
        c_h, c_w = int(H / 2), int(W / 2)  # 计算中心点距离
        per_h, per_w = H / (2 * block), W / (2 * block)  # 根据block的数量，等距计算1个block的宽高
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (block - c_h) * 2, W + (block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear',
                                          align_corners=True)  # 如果图片尺寸过小，需要插值放大。
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * block), W / (2 * block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整

        for i in range(block):
            i = i + 1
            if i < block:
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]  # 由中心向外扩张
                x_pre = None
                if i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)  # 扩大一圈，全部用0填充。
                    x_curr = x_curr - x_pad
                base_feature.append(x_curr)
            else:
                if i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                base_feature.append(x)

        x_0 = base_feature[0]
        x_1 = base_feature[1]
        x_2 = base_feature[2]
        x_3 = base_feature[3]

        x_4 = x_1 + F.pad(x_0, (per_h, per_h, per_w, per_w), "constant", 0)  # 第二阶段
        x_5 = x_2 + F.pad(x_1, (per_h, per_h, per_w, per_w), "constant", 0)
        x_6 = x_3 + F.pad(x_2, (per_h, per_h, per_w, per_w), "constant", 0)

        x_7 = x_2 + F.pad(x_4, (per_h, per_h, per_w, per_w), "constant", 0)  # 第三阶段
        x_8 = x_3 + F.pad(x_5, (per_h, per_h, per_w, per_w), "constant", 0)

        # x_9 = x_3 + F.pad(x_7, (per_h, per_h, per_w, per_w), "constant", 0)  # 第四阶段
        x_9 = x
        base_feature.append(x_4)
        base_feature.append(x_5)
        base_feature.append(x_6)
        base_feature.append(x_7)
        base_feature.append(x_8)
        base_feature.append(x_9)

        pixel_0 = x_0.size(2) * x_0.size(3)
        pixel_1 = x_1.size(2) * x_1.size(3) - x_0.size(2) * x_0.size(3)
        pixel_2 = x_2.size(2) * x_2.size(3) - x_1.size(2) * x_1.size(3)
        pixel_3 = x_3.size(2) * x_3.size(3) - x_2.size(2) * x_2.size(3)
        pixel_4 = pixel_0 + pixel_1
        pixel_5 = pixel_1 + pixel_2
        pixel_6 = pixel_2 + pixel_3
        pixel_7 = pixel_2 + pixel_4
        pixel_8 = pixel_3 + pixel_5
        pixel_9 = pixel_7 + pixel_3
        pixels_n.append(pixel_0)
        pixels_n.append(pixel_1)
        pixels_n.append(pixel_2)
        pixels_n.append(pixel_3)
        pixels_n.append(pixel_4)
        pixels_n.append(pixel_5)
        pixels_n.append(pixel_6)
        pixels_n.append(pixel_7)
        pixels_n.append(pixel_8)
        pixels_n.append(pixel_9)
        # for i in range(10):
        #     pixels_n.append(eval(f'pixel_{i}'))
        for j in range(len(base_feature)):
            x_curr = base_feature[j]
            if j == 0:
                x_pre = None
            else:
                x_pre = base_feature[j - 1]

            if pool1_gem == True:
                # res = self.gem_layers[i-1](x_curr, x_pre)
                name = 'gem' + str(i - 1)
                gem = getattr(self, name)
                # res = gem.gem(x_curr, x_pre,pix_num=pixels_n[j]) #Gem_new1_dense
                res = gem.gem(x_curr, x_pre)  # ！！Gem
                result.append(res)
            else:
                if j < 1:
                    avgpool = self.avg_pool(x_curr, pixels_n[j])
                else:
                    x_pre = base_feature[j - 1]
                    avgpool = self.avg_pool(x_curr, pixels_n[j])
                result.append(avgpool)
        return torch.stack(result, dim=2)

    def get_part_pool_dense_shift_new3(self, x, block=4, pool1_gem=False):
        # 软融合需要， 使用的时候需要解除注释
        #IN: [B,C,H,W] e.g.[2,2048,16,16]
        shift_p = self.shift_blocks(x).unsqueeze(-1).unsqueeze(-1)  # 可学习的偏移权重torch.Size([2, 3, 1, 1])
        pixels_n = []
        result_temp = [[], [], []]
        H, W = x.size(2), x.size(3)  # 计算input的宽高 H:16, W:16
        c_h, c_w = int(H / 2), int(W / 2)  # 计算中心点距离 c_h:8, c_w:8
        per_h, per_w = H / (2 * block), W / (2 * block)
        # 根据block的数量，等距计算1个block的宽高,即最小块的宽高以及每次取出的方环的宽度
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (block - c_h) * 2, W + (block - c_w) * 2 #per_h/per_w过小，插值至指定大小
            '''
            上述式子的意思是我得到c_h那么我至少需要扩展per_h/w到1。但看h，要扩展per_h到1，那么就至少要扩展h至2*block*per_h高
            '''
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear',
                                          align_corners=True)  # 如果图片尺寸过小，需要插值放大。
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * block), W / (2 * block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整

        #指定h,w方向的偏移常数
        shift_h = [-2, 0, 2]
        shift_w = [-2, 0, 2]
        for j in range(len(shift_h)):
            base_feature = []
            for i in range(block):  #一个block一个block的来扩展其特征图
                i = i + 1
                if i < block:
                    x_curr = x[:, :, (c_h - i * per_h + shift_h[j]):(c_h + i * per_h + shift_h[j]),
                             (c_w - i * per_w + shift_w[j]):(c_w + i * per_w + shift_w[j])]  # 由中心向外扩张
                    x_pre = None
                    if i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)  # 扩大一圈，全部用0填充。
                        x_curr = x_curr - x_pad
                    base_feature.append(x_curr)
                else:
                    if i > 1:
                        x_pre = x[:, :, (c_h - (i - 1) * per_h + shift_h[j]):(c_h + (i - 1) * per_h + shift_h[j]),
                                (c_w - (i - 1) * per_w + shift_w[j]):(c_w + (i - 1) * per_w + shift_w[j])]
                        pad_h = c_h - (i - 1) * per_h
                        pad_w = c_w - (i - 1) * per_w
                        # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                        if x_pre.size(2) + 2 * pad_h == H:
                            x_pad = F.pad(x_pre, (
                            pad_w + shift_w[j], pad_w - shift_w[j], pad_h + shift_h[j], pad_h - shift_h[j]), "constant",
                                          0)
                        else:
                            ep = H - (x_pre.size(2) + 2 * pad_h)
                            x_pad = F.pad(x_pre, (
                            pad_w + ep + shift_w[j], pad_w - shift_w[j], pad_h + ep + shift_h[j], pad_h - shift_h[j]),
                                          "constant", 0)
                        x = x - x_pad
                    base_feature.append(x)
            '''
            1 [2,2048,4,4]
            2 [2,2048,8,8]-[中心块置0]；
            3 [2,2048,12,12]-[中心[2,2048,8,8]部分置0]
            4 [2,2048,16,16]-[中心[2,2048,12,12]部分置0]
            然后依次填充为论文中三角形
            如此重复3次(不同偏移量)
            '''
            x_0 = base_feature[0]
            x_1 = base_feature[1]
            x_2 = base_feature[2]
            x_3 = base_feature[3]

            x_4 = x_1 + F.pad(x_0, (per_h, per_h, per_w, per_w), "constant", 0)  # 第二阶段
            x_5 = x_2 + F.pad(x_1, (per_h, per_h, per_w, per_w), "constant", 0)
            x_6 = x_3 + F.pad(x_2, (per_h, per_h, per_w, per_w), "constant", 0)

            x_7 = x_2 + F.pad(x_4, (per_h, per_h, per_w, per_w), "constant", 0)  # 第三阶段
            x_8 = x_3 + F.pad(x_5, (per_h, per_h, per_w, per_w), "constant", 0)

            # x_9 = x_3 + F.pad(x_7, (per_h, per_h, per_w, per_w), "constant", 0)  # 第四阶段
            x_9 = x

            base_feature.append(x_4)
            base_feature.append(x_5)
            base_feature.append(x_6)
            base_feature.append(x_7)
            base_feature.append(x_8)
            base_feature.append(x_9)

            pixel_0 = x_0.size(2) * x_0.size(3)
            pixel_1 = x_1.size(2) * x_1.size(3) - x_0.size(2) * x_0.size(3)
            pixel_2 = x_2.size(2) * x_2.size(3) - x_1.size(2) * x_1.size(3)
            pixel_3 = x_3.size(2) * x_3.size(3) - x_2.size(2) * x_2.size(3)
            pixel_4 = pixel_0 + pixel_1
            pixel_5 = pixel_1 + pixel_2
            pixel_6 = pixel_2 + pixel_3
            pixel_7 = pixel_2 + pixel_4
            pixel_8 = pixel_3 + pixel_5
            pixel_9 = pixel_7 + pixel_3
            pixels_n.append(pixel_0)
            pixels_n.append(pixel_1)
            pixels_n.append(pixel_2)
            pixels_n.append(pixel_3)
            pixels_n.append(pixel_4)
            pixels_n.append(pixel_5)
            pixels_n.append(pixel_6)
            pixels_n.append(pixel_7)
            pixels_n.append(pixel_8)
            pixels_n.append(pixel_9)
            # for i in range(10):
            #     pixels_n.append(eval(f'pixel_{i}'))
            for k in range(len(base_feature)):
                x_curr = base_feature[k]
                if k == 0:
                    x_pre = None
                else:
                    x_pre = base_feature[k - 1]

                if pool1_gem == True:
                    # res = self.gem_layers[i-1](x_curr, x_pre)
                    name = 'gem' + str(k)
                    gem = getattr(self, name)
                    res = gem.gem(x_curr, x_pre)
                    result_temp[j].append(res)
                else:
                    if k < 1:
                        avgpool = self.avg_pool(x_curr, pixels_n[k])
                    else:
                        x_pre = base_feature[k]
                        avgpool = self.avg_pool(x_curr, pixels_n[k])
                    result_temp[j].append(avgpool)

        result_0 = torch.stack(result_temp[0], dim=2)
        result_1 = torch.stack(result_temp[1], dim=2)
        result_2 = torch.stack(result_temp[2], dim=2)
        # 软融合
        result = torch.cat((result_0.unsqueeze(1), result_1.unsqueeze(1), result_2.unsqueeze(1)), dim=1)
        result = torch.sum((result * shift_p), dim=1)  # 使用的时候需要解除注释
        return result

    def part_classifier(self, x, cls_name='classifier'):
        # [B, dim, C]
        part = {}
        predict = {}
        for i in range(x.shape[-1]):
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
        for i in range(x.shape[-1]):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

    def avg_pool(self, x_curr, pixel_n=None):
        h, w = x_curr.size(2), x_curr.size(3)
        # pix_num = h * w - h_pre * w_pre
        pix_num = pixel_n
        avg = x_curr.flatten(start_dim=2).sum(dim=2).div_(pix_num)
        return avg

class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim=2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) #initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, x_pre=None, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1) # torch.Size([2, 8, 8, 2048])
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1) # torch.Size([2, 2048, 8, 8])
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1))) # torch.Size([2, 2048, 1, 1])
        x = x.view(x.size(0), x.size(1)) # torch.Size([2, 2048])
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'


def main():
    x = torch.randn(2, 1024, 12,12)
    m = SDPL_Module(num_classes=701, pool='avg', num_blocks=4, feature_dim=1024)
    z = m(x)
    # print(m)
    # print_nb_params(m)
    # evaluate_model(m,x)
    print(f'Input shape is {x.shape}')
    # if isinstance(z,tuple):
    #     print(f'Training Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    # else:
    #     print(f'Training Output shape is {torch.stack(z,dim=0).shape}')

    m.eval()
    z=m(x)
    evaluate_model(m,x)
    if isinstance(z,tuple):
        print(f'Evaluating Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    else:
        print(f'Evaluating Output shape is {z.shape}')

if __name__ == "__main__":
    main()