import torch
import torch.nn as nn
from model.components.FSRA_component import ClassBlock
from utils.commons import print_nb_params,evaluate_model
class FSRA_Module(nn.Module):
    def __init__(self, num_classes, num_blocks, feature_dim, return_f=False):
        super(FSRA_Module, self).__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.return_f = return_f
        self.feature = torch.Tensor()
        self.classifier1 = ClassBlock(feature_dim, num_classes, 0.5, return_f=return_f)
        for i in range(self.num_blocks):
            name = 'classifier_heat' + str(i + 1)
            setattr(self, name, ClassBlock(feature_dim, num_classes, 0.5, return_f=return_f))

    def forward(self, x):
        #[B, patch_size + 1, dim]
        # x = torch.einsum('bcd->bdc',torch.cat([x[0].unsqueeze(-1),x[1].flatten(2)],dim=-1))
        transformer_feature = self.classifier1(x[:, 0])

        if self.num_blocks == 1:
            return transformer_feature

        part_features = x[:, 1:]
        heat_result = self.get_heatmap_pool(part_features)
        y = self.part_classifier(self.num_blocks, heat_result, cls_name='classifier_heat')

        if self.training:
            y = y + [transformer_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            transformer_feature = transformer_feature.view(transformer_feature.size(0), -1, 1)
            y = torch.cat([y, transformer_feature], dim=2)
        return y

    def get_heatmap_pool(self, part_features, add_global=False, otherbranch=False):
        # [B, patch_size, dim] --> [B, patch_size]
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)  # patch_size
        # argsort([B, patch_size],desc=True)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        # [B, patch_size, dim](patch token) --> list(B,[patch_size, dim]) (对每一个dim的patch进行重新排序，然后)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]  # 使用每一batch中每一样本对应的排序结果进行排序
        # 重新变为tensor：list(B,[patch_size, dim])-->[B, patch_size, dim](patch token)
        x_sort = torch.stack(x_sort, dim=0)
        self.feature = x_sort.detach()
        # 这一段就是分隔开n个类别
        split_each = size / self.num_blocks
        # 这个是给不同的类别提供长度值
        split_list = [int(split_each) for i in range(self.num_blocks - 1)]
        split_list.append(size - sum(split_list))
        # 将[B, patch_size, dim]-->[c_num,[B,patch_size//c_num,dim]]
        split_x = x_sort.split(split_list, dim=1)
        # 求得每一个patch对应的类别的均值[c_num,[B, dim]]
        split_list = [torch.mean(split, dim=1) for split in split_x]
        # [c_num,[B, dim]]-->[B, dim, c_num]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            #
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1,
                                                                                                     self.num_blocks)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, num_blocks, x, cls_name='classifier_lpn'):
        # [B, dim, C]
        part = {}
        predict = {}
        for i in range(num_blocks):
            # [B, dim, 1]-->[B, dim]
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            # 调用transformer中的特定模块进行预测，如classifier_heat1
            name = cls_name + str(i + 1)
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
    # x = torch.randn(20, 401, 768)
    out_dim = 768
    x = [torch.randn(5, out_dim), torch.randn(5, out_dim, 12, 12)]
    m = FSRA_Module(num_classes=701, num_blocks=3, feature_dim=768, return_f=False)
    z = m(x)
    # z = torch.stack(z,dim=0)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    if isinstance(z,tuple):
        print(f'Training Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    else:
        print(f'Training Output shape is {torch.stack(z,dim=0).shape}')

    m.train(False)
    z = m(x)
    if isinstance(z,tuple):
        print(f'Evaluating Output shape is cls:{torch.stack(z[0],dim=0).shape},feature:{torch.stack(z[1],dim=0).shape}')
    else:
        print(f'Evaluating Output shape is {z.shape}')

    evaluate_model(m,x)

if __name__ == "__main__":
    main()