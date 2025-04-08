import torch
from model import components
from model import backbones

def get_backbone(backbone_arch=None,backbone_configs=None):
    if 'resnet' in backbone_arch.lower():
        assert 'pretrained' in backbone_configs
        assert 'layers_to_freeze' in backbone_configs
        assert 'layers_to_crop' in backbone_configs
        return backbones.ResNet(backbone_arch, **backbone_configs)
    elif 'vit_fsra' in backbone_arch.lower():
        assert 'img_size' in backbone_configs
        assert 'return_token' in backbone_configs
        assert 'adapter' in backbone_configs
        assert 'layers_to_freeze' in backbone_configs
        m = backbones.vit_small_patch16_224_FSRA(**backbone_configs)
        # m = backbones.vit_base_patch16_224_FSRA(**backbone_configs)
        try:
            # m.load_param('/home/whu/Documents/codespace/FSRA/models/vit_small_p16_224-15ec54c9.pth')
            m.load_param('/media/whu/Filesystem2/jx_vit_base_p16_224-80ecf9dd.pth')
            # print("successfully load pretrained model!")
        except:
            raise KeyError
        return m
    elif 'van' in backbone_arch.lower():
        assert 'pretrained_path' in backbone_configs
        assert 'layers_to_freeze' in backbone_configs
        assert 'layers_to_crop' in backbone_configs
        # backbone = van_small()
        # checkpoint = torch.load(pretrain_path)["state_dict"]
        # backbone.load_state_dict(checkpoint)
        raise UserWarning('this model DID NOT LOAD  pretrained model, please edit by yourself in function')
        # return backbones.van_small(backbone_arch, **backbone_configs)
    elif 'dinov2' in backbone_arch.lower():
        assert 'layers_to_freeze' in backbone_configs
        assert 'layers_to_crop' in backbone_configs
        assert 'norm_layer' in backbone_configs
        assert 'return_token' in backbone_configs
        return backbones.DINOv2(model_name=backbone_arch, **backbone_configs)
    elif 'convnext' in backbone_arch.lower():
        assert 'stages_to_freeze' in backbone_configs
        assert 'return_token' in backbone_configs
        assert 'return_chunk' in backbone_configs
        return backbones.ConvNeXt(model_name=backbone_arch, **backbone_configs)
    elif 'transnext' in backbone_arch.lower():
        assert 'layers_to_freeze' in backbone_configs
        assert 'return_token' in backbone_configs
        assert 'return_chunk' in backbone_configs
        assert 'adapter' in backbone_configs
        return backbones.TransNeXt(model_name=backbone_arch, **backbone_configs)
    elif 'swinv2' in backbone_arch.lower():
        return backbones.Swinv2(model_name=backbone_arch)
    else:
        raise KeyError('incorrect backbone selection')

def get_components(compo_arch=None, compo_configs=None):
    if 'fsra' in compo_arch.lower():
        assert 'num_classes' in compo_configs
        assert 'num_blocks' in compo_configs
        assert 'feature_dim' in compo_configs
        assert 'return_f' in compo_configs
        return components.FSRA_Module(**compo_configs)
    elif 'lpn' in compo_arch.lower():
        assert 'num_classes' in compo_configs
        assert 'pool' in compo_configs
        assert 'num_blocks' in compo_configs
        assert 'feature_dim' in compo_configs
        assert 'return_f' in compo_configs
        return components.LPN_Module(**compo_configs)
    elif 'sdpl' in compo_arch.lower():
        assert 'num_classes' in compo_configs
        assert 'pool' in compo_configs
        assert 'num_blocks' in compo_configs
        assert 'feature_dim' in compo_configs
        return components.SDPL_Module(**compo_configs)
    elif 'qdfl' in compo_arch.lower():
        assert 'in_channels' in compo_configs
        assert 'num_classes' in compo_configs
        assert 'num_supervisor' in compo_configs
        assert 'query_configs' in compo_configs
        return components.QDFL(**compo_configs)
    elif 'ccr' in compo_arch.lower():
        assert 'in_channels' in compo_configs
        assert 'num_classes' in compo_configs
        assert 'block' in compo_configs
        assert 'M' in compo_configs
        assert 'return_f' in compo_configs
        return components.CCR(**compo_configs)
    elif 'mccg' in compo_arch.lower():
        assert 'in_channels' in compo_configs
        assert 'num_classes' in compo_configs
        assert 'block' in compo_configs
        assert 'return_f' in compo_configs
        return components.MCCG(**compo_configs)
    elif 'dac' in compo_arch.lower():
        assert 'num_classes' in compo_configs
        assert 'block' in compo_configs
        assert 'return_f' in compo_configs
        return components.DAC(**compo_configs)
    elif 'gem' == compo_arch.lower():
        assert 'in_channels' in compo_configs
        assert 'cls_token' in compo_configs
        assert 'num_classes' in compo_configs
        assert 'return_f' in compo_configs
        return components.GeM_Baseline(**compo_configs)
    elif 'netvlad' in compo_arch.lower():
        assert 'num_clusters' in compo_configs
        assert 'dim' in compo_configs
        assert 'normalize_input' in compo_configs
        assert 'vladv2' in compo_configs
        return components.NetVLAD(**compo_configs)
    else:
        raise KeyError(f"{compo_arch} not implemented")

