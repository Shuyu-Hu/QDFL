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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or 2 * in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.fc2(x)

        return x

class GeMPool(nn.Module):
    def __init__(self, p=4.5, eps=1e-7):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # [B C H W]
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False,KD=False):
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
        self.KD = KD

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


class CycleFC(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels * 2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0

        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)


class CoarseQuery(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(CoarseQuery, self).__init__()
        self.self_attn0 = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_x = torch.nn.LayerNorm(in_dim)

        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))
        nn.init.normal_(self.queries, std=1e-6)
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        try:
            B, _, _ = x.shape
        except:
            B, _, _, _ = x.shape
            x = einops.rearrange(x, 'i j k l->i (k l) j')

        x = self.norm_x(x + self.self_attn0(x, x, x)[0])

        q = self.queries.repeat(B, 1, 1)

        q = self.norm_q(q + self.self_attn(q, q, q)[0])

        out = self.norm_out(self.cross_attn(q, x, x)[0])
        return x, out

class FFU(torch.nn.Module):
    def __init__(self, linear_out_dim):
        super(FFU, self).__init__()
        self.linear_out_dim = linear_out_dim
        self.cycle1 = CycleFC(self.linear_out_dim, self.linear_out_dim, (1, 3), 1, 0)
        self.cycle2 = CycleFC(self.linear_out_dim, self.linear_out_dim, (3, 1), 1, 0)
        self.reweight = MLP(self.linear_out_dim, self.linear_out_dim//3, self.linear_out_dim*3)

    def forward(self,x):
        x1 = self.cycle1(x)
        x2 = self.cycle2(x.permute(0, 1, 3, 2))

        B, C, H, W = x.shape
        a = (x1 + x2 + x).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)
        z = x1.permute(0, 2, 3, 1) * a[0] + x2.permute(0, 2, 3, 1) * a[1] + x.permute(0, 2, 3, 1) * a[2]
        return z.permute(0, 3, 1, 2) + x

class AQEU(torch.nn.Module):
    def __init__(self, in_channels, linear_out_dim, num_queries, num_supervisor):
        super(AQEU,self).__init__()
        self.linear_out_dim = linear_out_dim
        self.num_queries = num_queries
        self.num_supervisor = num_supervisor
        # self.fc = torch.nn.Conv2d(in_channels, self.linear_out_dim, kernel_size=(1, 1))
        self.fc = torch.nn.Conv2d(in_channels, self.linear_out_dim, kernel_size=(1, 1))
        # self.fc = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=self.linear_out_dim, kernel_size=1)
        self.coarse = CoarseQuery(self.linear_out_dim, self.num_queries, nheads=8)
        self.FFU = FFU(self.linear_out_dim)
        self.norm_q = torch.nn.LayerNorm(self.linear_out_dim)
        self.cross_attn = torch.nn.MultiheadAttention(self.linear_out_dim, num_heads=8, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(self.linear_out_dim)
        # self.dropout = nn.Dropout(p=0.5)
        # for m in self.modules():
        #     if isinstance(m, (nn.Linear)):
        #         nn.init.trunc_normal_(m.weight, std=0.02)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        self.query_remap = nn.Linear(self.num_queries, self.num_supervisor)

    def forward(self,x):
        x = self.fc(x)
        # x = self.dropout(x)
        x = einops.rearrange(x, 'i j k l->i (k l) j')
        x_coarse, q_coarse = self.coarse(x)
        x_coarse = x_coarse + x
        x_coarse = einops.rearrange(x_coarse, 'b (h w) c -> b c h w', h=int(math.sqrt(x_coarse.shape[1])), w=int(math.sqrt(x_coarse.shape[1])))
        x_fine_0 = self.FFU(x_coarse)
        x_fine = einops.rearrange(x_fine_0, 'b c h w -> b (h w) c')
        q_coarse = self.norm_q(q_coarse)

        q_fine = q_coarse + self.cross_attn(q_coarse, x_fine, x_fine)[0]
        q_fine = self.norm_out(q_fine)

        y = self.query_remap(q_fine.permute(0, 2, 1))
        return y, x_fine_0

class QDFL(torch.nn.Module):
    def __init__(self, in_channels, num_classes, cls_token, query_configs, num_supervisor, return_f=False, KD=False):
        super(QDFL, self).__init__()
        self.in_channels = in_channels
        self.linear_out_dim = 512
        self.num_classes = num_classes
        self.num_queries = query_configs['num_queries']
        self.num_supervisor = num_supervisor
        self.return_f = return_f
        self.cls_token = cls_token
        self.KD = KD

        self.AQEU = AQEU(self.in_channels,self.linear_out_dim,self.num_queries,self.num_supervisor)
        self.gem_fusion = GeMPool()

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cls_token:
            self.token_classifier = ClassBlock(self.in_channels, num_classes, 0.5, return_f=return_f,KD=self.KD)

        self.gem_feature_classifier = ClassBlock(self.linear_out_dim, num_classes, 0.5, return_f=return_f,KD=self.KD)
        for i in range(2):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(self.linear_out_dim, self.num_classes, 0.5, return_f=return_f,KD=self.KD))

    def forward(self, x):
        if self.cls_token:
            t, x = x
            token_feature = self.token_classifier(t)
        try:
            B, _, _ = x.shape
            if x.shape[-1] == self.in_channels:
                x = x.permute(0,2,1)
            x = x.reshape(B,self.in_channels,int(math.sqrt(int(x.shape[-1]))),int(math.sqrt(x.shape[-1])))
        except:
            B, _, _, _ = x.shape

        aqeu_feature, x_fine = self.AQEU(x)

        gem_feature = self.gem_fusion(x_fine)
        fusion_feature = self.gem_feature_classifier(gem_feature)

        y = self.part_classifier(aqeu_feature, cls_name='classifier')

        if self.cls_token:
            if self.training or self.KD:
                # y = [fusion_feature]
                y = y + [token_feature] + [fusion_feature]
                if self.return_f:  # Use metric learning or not
                    cls, features = [], []
                    for i in y:
                        cls.append(i[0])
                        features.append(i[1])
                    return cls, features
                else:
                    return y, None
            else:
                token_feature = token_feature.view(token_feature.size(0), -1, 1)
                fusion_feature = fusion_feature.view(fusion_feature.size(0), -1, 1)
                y = torch.cat([y, token_feature, fusion_feature], dim=2)
        else:
            if self.training or self.KD:
                y = y + [fusion_feature]
                if self.return_f:
                    cls, features = [], []
                    for i in y:
                        cls.append(i[0])
                        features.append(i[1])
                    return cls, features
                else:
                    return y, None
            else:
                y = torch.cat([y, fusion_feature.unsqueeze(-1)], dim=2)
        # return y, x_fine
        return y

    def part_classifier(self, x, cls_name='classifier'):
        # [B, dim, num_feature]
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
        if not self.training and not self.KD:
            return torch.stack(y, dim=2)
        return y


def main():
    out_dim = 768
    x = [torch.randn(5, out_dim), torch.randn(5, out_dim, 12, 12)]
    # x = torch.randn(2, out_dim, 32, 32)  # b,c,h,w
    boq_configs = {
        'num_queries': 8
    }
    agg = QDFL(in_channels=out_dim, num_classes=701, cls_token=True, return_f=True,
                        query_configs=boq_configs, num_supervisor=2).train()
    print(agg)
    print_nb_params(agg)
    x_out = agg(x)

    # from utils.commons import evaluate_model
    # evaluate_model(agg, x, 0)
    # print(x_out.shape)

    from torchinfo import summary

    summary(agg, input_data=x)


if __name__ == '__main__':
    main()
