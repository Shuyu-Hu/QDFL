import time
import einops
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from model.backbones.convnext_backbone import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge
from utils.commons import print_nb_params, evaluate_model
ConvNeXt_ARCHS = {
    'convnext_tiny': [96, 192, 384, 768],
    'convnext_small': [96, 192, 384, 768],
    'convnext_base': [128, 256, 512, 1024],
    'convnext_large': [192, 384, 768, 1536],
    'convnext_xlarge': [256, 512, 1024, 2048],
}

class ConvNeXt(nn.Module):
    def __init__(
            self,
            model_name,
            stages_to_freeze,
            return_token,
            return_chunk=False
    ):
        super().__init__()

        assert model_name in ConvNeXt_ARCHS.keys(), f'Unknown model_dino name {model_name}'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # this path is hardcoded, since this work has changed comparing with the original ConvNeXt code
        # self.model_dino = torch.hub.load('/home/whu/Documents/codespace/mixvpr/MixVPR/facebookresearch_ConvNeXt_main/', model_name,trust_repo=True,source='local').to(self.device)
        self.model = self.load_convnext(model_name).to(self.device)
        self.num_channels = ConvNeXt_ARCHS[model_name][-1]
        self.max_stage = 4
        self.stages_to_freeze = stages_to_freeze
        self.return_token = return_token
        self.tlist = []

        assert self.stages_to_freeze <= self.max_stage, 'Please check whether the layer number is correct'
        if self.stages_to_freeze != 0:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            for i, blk in enumerate(self.model.stages):
                if self.stages_to_freeze <= i < self.max_stage:
                    for param in blk.parameters():
                        param.requires_grad = True

        self.return_chunk = return_chunk

    def load_convnext(self, model_name):
        if model_name.lower() == 'convnext_tiny':
            model = convnext_tiny()
            
        elif model_name.lower() == 'convnext_small':
            model = convnext_small()
            
        elif model_name.lower() == 'convnext_base':
            model = convnext_base()

        elif model_name.lower() == 'convnext_large':
            model = convnext_large()

        elif model_name.lower() == 'convnext_xlarge':
            model = convnext_xlarge()
        else:
            return None

        return model

    def forward_features(self, x):
        x_chunk = []
        for i in range(4):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            if self.return_chunk:
                x_chunk.append(x)
        if self.return_chunk:
            return x_chunk
        return self.model.norm(x.mean([-2, -1])), x

    def forward(self, x):
        x = self.forward_features(x)
        if self.return_chunk:
            return x
        if self.return_token:
            return x
        return x[1]


def main():
    H = 384
    x = torch.randn(20, 3, H, H).to(device='cuda')
    m = ConvNeXt(model_name='convnext_base',
               stages_to_freeze=3,
               return_token=False,
               return_chunk=True).to(device='cuda')
    start = time.time()
    z = m(x)
    end = time.time()
    print(m)
    print_nb_params(m, 'adapters')
    # print(f'Input shape is {x.shape}')
    # print(f'Output shape is {z.shape}')
    print(f'Infer time is {end - start}')
    evaluate_model(m,x)

if __name__ == '__main__':
    main()
