import time

import einops
import torch
import torch.nn as nn
from utils.commons import print_nb_params,evaluate_model
from model.backbones.dinov2_backbone import vit_small, vit_base, vit_large, vit_giant2

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}
DINOV2_WEIGHTS = {
    'dinov2_vits14': '/home/whu/Documents/codespace/mixvpr/MixVPR/models/pretrained_weights/dinov2_vits14_pretrain.pth',
    'dinov2_vitb14': '/home/whu/Documents/codespace/mixvpr/MixVPR/models/pretrained_weights/dinov2_vitb14_pretrain.pth',
    'dinov2_vitl14': '/home/whu/Documents/codespace/mixvpr/MixVPR/models/pretrained_weights/dinov2_vitl14_pretrain.pth',
    'dinov2_vitg14': 'none',
}
DINOV2_LAYERS = {
    'dinov2_vits14': 12,
    'dinov2_vitb14': 12,
    'dinov2_vitl14': 24,
    'dinov2_vitg14': 40,
}


class SideAdapter(nn.Module):  # Adapter is used to add to the transformer block for global adaptation
    def __init__(self, D_features, D_hidden_features=16, act_layer=nn.GELU, skip_connect=True, alpha=0.5):
        # mlp_ratio is the bottleneck ratio of adapters
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.alpha = alpha

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + self.alpha * xs
        else:
            x = xs
        return x


class DINOv2(nn.Module):
    """
    DINOv2 model_dino

    Args:
        model_name (str): The name of the model_dino architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        layers_to_freeze (int): The number of last blocks in the model_dino that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
            self,
            model_name,
            layers_to_freeze,
            layers_to_crop,
            norm_layer,
            return_token,
            return_token_list,
            adapter=None,
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model_dino name {model_name}'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # this path is hardcoded, since this work has changed comparing with the original dinov2 code
        # self.model_dino = torch.hub.load('/home/whu/Documents/codespace/mixvpr/MixVPR/facebookresearch_dinov2_main/', model_name,trust_repo=True,source='local').to(self.device)
        self.model = self.load_dino(model_name).to(self.device)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.max_layer = DINOV2_LAYERS[model_name]
        self.layers_to_freeze = layers_to_freeze
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.return_token_list = return_token_list
        self.layers_to_crop = layers_to_crop
        self.adapter = adapter

        assert self.layers_to_freeze <= self.max_layer, 'Please check whether the layer number is correct'
        if self.layers_to_freeze >= 0:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            for i, blk in enumerate(self.model.blocks):
                if self.layers_to_freeze <= i < self.max_layer:
                    for param in blk.parameters():
                        param.requires_grad = True

        # Ensure adapter parameters are trainable if adapter is used
        if self.adapter is not None:
            for name, module in self.model.named_modules():
                if 'adapter' in name:
                    for param in module.parameters():
                        param.requires_grad = True

        for layer_index in self.layers_to_crop:
            if layer_index <= self.max_layer:
                self.model.blocks[layer_index] = nn.Identity()
            else:
                raise ValueError("Layer index should not exceed 0-{}.".format(self.max_layer))

        # for n, m in self.named_modules():
        #     if 'adapter' in n:
        #         for n2, m2 in m.named_modules():
        #             if 'D_fc2' in n2:
        #                 if isinstance(m2, nn.Linear):
        #                     nn.init.constant_(m2.weight, 0.)
        #                     nn.init.constant_(m2.bias, 0.)

    def load_dino(self, model_name):
        if model_name.lower() == 'dinov2_vits14':
            model = vit_small(patch_size=14, img_size=518, init_values=1, block_chunks=0)

        elif model_name.lower() == 'dinov2_vitb14':
            model = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)

        elif model_name.lower() == 'dinov2_vitl14':
            model = vit_large(patch_size=14, img_size=518, init_values=1, block_chunks=0)

        elif model_name.lower() == 'dinov2_vitg14':
            model = vit_giant2(patch_size=14, img_size=518, init_values=1, block_chunks=0)

        model_dict = model.state_dict()
        state_dict = torch.load(DINOV2_WEIGHTS[model_name])
        model_dict.update(state_dict.items())
        model.load_state_dict(model_dict, strict=True)

        return model

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        for blk in self.model.blocks:
            x = blk(x)
            if self.return_token_list:
                self.tlist.append(x[:, 0, :])

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels))

        if self.return_token_list:
            self.tlist.append(t)

        if self.return_token_list:
            t_list = torch.stack(self.tlist, dim=0)
            self.tlist = []
            return f, t_list

        if self.return_token:
            return t,einops.rearrange(f,'b h w c -> b c h w')

        return einops.rearrange(f,'b h w c -> b c h w')



def main():
    H = 224
    x = torch.randn(20, 3, H, H).to(device='cuda')
    m = DINOv2(model_name='dinov2_vitb14',
               layers_to_freeze=0,
               layers_to_crop=[],
               norm_layer=True,
               return_token=False,
               return_token_list=False,
               adapter=False
               )
    start = time.time()
    z = m(x)
    end = time.time()
    print(m)
    print_nb_params(m, 'adapter')
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {z.shape}')
    print(f'Infer time is {end - start}')
    # from utils.commons import evaluate_model
    # evaluate_model(m, x, 0)

    from torchinfo import summary

    summary(m, input_data=x)


if __name__ == '__main__':
    main()
