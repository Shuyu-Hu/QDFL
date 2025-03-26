import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_t,swin_v2_s,swin_v2_b
from utils import print_nb_params
from utils.commons import evaluate_model


SWINV2_ARCHS = {
    'swinv2_tiny': 'swinv2_tiny_window16_256',
    'swinv2_small': 'swinv2_small_window16_256',
    'swinv2_base': 'swinv2_base_window8_256'
}

class Swinv2(nn.Module):
    def __init__(self,
                 model_name):
        super(Swinv2, self).__init__()
        assert model_name in SWINV2_ARCHS.keys(), f'Unknown model_swin name {model_name}'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 创建 Swin Transformer 模型，并加载预训练权重
        self.model = self.load_swin(model_name)

    def load_swin(self,model_name):
        if model_name.lower() == 'swinv2_tiny':
            model = swin_v2_t(weights='IMAGENET1K_V1')

        elif model_name.lower() == 'swinv2_small':
            model = swin_v2_s(weights='IMAGENET1K_V1')

        elif model_name.lower() == 'swinv2_base':
            model = swin_v2_b(weights='IMAGENET1K_V1')
        else:
            raise KeyError('model_name not found!')
        model.avgpool = None
        model.flatten = None
        model.head = None
        return model

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        return x
def main():
    H = 256
    x = torch.randn(20, 3, H, H).to(device='cuda')
    m = Swinv2(model_name='swinv2_base').to(device='cuda')
    start = time.time()
    z = m(x)
    print(z)
    end = time.time()
    # print(m)
    # print_nb_params(m, 'adapters')
    # print(f'Input shape is {x.shape}')
    # print(f'Output shape is {z.shape}')
    print(f'Infer time is {end - start}')
    evaluate_model(m,x)

if __name__ == '__main__':
    main()

