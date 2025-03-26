import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision
from numpy.matlib import empty


class ResNet(nn.Module):
    def __init__(self,
                 model_name,
                 pretrained,
                 layers_to_freeze,
                 layers_to_crop,
                 change_stride=False,
                 return_token=False,
                 chunk=False,
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze
        self.change_stride = change_stride
        self.return_token = return_token
        self.chunk = chunk
        self.layers_to_crop = layers_to_crop if not self.chunk else []

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model_dino = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model_dino is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)
            if layers_to_freeze >= 4:
                self.model.layer4.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in self.layers_to_crop:
            self.model.layer4 = None
        if 3 in self.layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512

        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        if self.change_stride is True:
            self.model.layer4[0].downsample[0].stride = (1, 1)
            self.model.layer4[0].conv2.stride = (1, 1)

        if self.return_token:
            self.norm = nn.LayerNorm(self.out_channels, eps=1e-6)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        if self.model.layer3 is not None:
            x3 = self.model.layer3(x2)
        else:
            x3 = x2
        if self.model.layer4 is not None:
            x4 = self.model.layer4(x3)
        else:
            x4 = x3
        # return einops.rearrange(x,'i j k l->i (k l) j')
        if self.return_token:
            return self.norm(x4.mean([-2, -1])), x4

        if self.chunk:
            return [x1,x2,x3,x4]

        return x4


def print_nb_params(m, find_name=None):
    name = type(m).__name__  # Get the class name of the module
    max_param_size = None
    max_param_count = 0
    max_param_name = None

    # Filter model_dino parameters that require gradients
    model_parameters = list(filter(lambda p: p.requires_grad, m.parameters()))
    params_sizes = [(p.size(), np.prod(p.size())) for p in model_parameters]

    # Debugging: Print the filtered parameters
    print("Filtered parameters:")
    for p in model_parameters:
        print(f"Parameter: {p.size()}, requires_grad: {p.requires_grad}")

    # Check if params_sizes is populated
    if not params_sizes:
        print("Warning: params_sizes is empty. No trainable parameters found.")
        return

    total_params = sum([size for _, size in params_sizes])
    flag = True

    for sub_name, sub_module in m.named_modules():
        if isinstance(sub_module, nn.Linear):
            if not params_sizes:
                print(f"Warning: No more params to pop for module {sub_name}")
                break

            param_size, param_count = params_sizes.pop(0)
            if param_count > max_param_count:
                max_param_size = param_size
                max_param_count = param_count
                max_param_name = sub_name
        if find_name is not None and find_name in sub_name:
            for param in sub_module.parameters():
                flag = flag * param.requires_grad
        else:
            flag = False
    print(bool(flag))

    print(f'{name} - Trainable parameters: {total_params / 1e6:.3f}M')
    print(f"{name} - Maximum parameter size:", max_param_size)
    print(f"{name} - Maximum parameter count:", max_param_count)
    print(f"{name} - Maximum parameter name:", max_param_name)


def main():
    x = torch.randn(1, 3, 224,224)
    m = ResNet(model_name='resnet34',
               pretrained=True,
               layers_to_freeze=0,
               return_token=False,
               layers_to_crop=[3],
               chunk=True)
    r = m(x)
    print(m)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')
    print(list(m.parameters())[-1].shape[0])


if __name__ == '__main__':
    main()
