import torch
import yaml
import time
import numpy as np
import torch.nn as nn
from ptflops import get_model_complexity_info
from pytorch_model_summary import summary
from thop import profile, clever_format


def print_nb_params(m, find_name=None):
    name = type(m).__name__  # Get the class name of the module
    max_param_size = None
    max_param_count = 0
    max_param_name = None

    # Filter model_dino parameters that require gradients
    model_parameters = list(filter(lambda p: p.requires_grad, m.parameters()))
    params_sizes = [(p.size(), np.prod(p.size())) for p in model_parameters]

    # Check if params_sizes is populated
    if not params_sizes:
        print("Warning: params_sizes is empty. No trainable parameters found.")
        return

    total_params = sum([size for _, size in params_sizes])

    found = False
    for sub_name, sub_module in m.named_modules():
        if isinstance(sub_module, (nn.Linear, nn.Conv2d)):
            if not params_sizes:
                print(f"Warning: No more params to pop for module {sub_name}")
                break

            param_size, param_count = params_sizes.pop(0)
            if param_count > max_param_count:
                max_param_size = param_size
                max_param_count = param_count
                max_param_name = sub_name

        # Check if the layer's name matches the find_name
        if find_name is not None and find_name in sub_name:
            found = True
            print(f"Layer name: {sub_name}")

            # Check if the sub_module has any parameters
            params = list(sub_module.parameters())
            if params:
                print(f"Parameter size: {np.prod(params[0].size()) / 1e6:.3f}M")
                print(f"Requires gradients: {params[0].requires_grad}")
            else:
                print("This layer has no parameters.")

    print(f'{name} - Trainable parameters: {total_params / 1e6:.3f}M')
    print(f"{name} - Maximum parameter size:", max_param_size)
    print(f"{name} - Maximum parameter count:", max_param_count)
    print(f"{name} - Maximum parameter name:", max_param_name)

    if not found and find_name is not None:
        print(f"Layer '{find_name}' not found in the model.")


def evaluate_model(model, input_tensor, device=0):
    def get_shape_or_length(x):
        if isinstance(x, list):
            return [get_shape_or_length(sub_x) for sub_x in x]
        elif hasattr(x, 'shape'):
            return x.shape
        else:
            return type(x)

    # 打印模型参数信息
    print_nb_params(model)

    with torch.cuda.device(device):
        # 计算模型复杂度信息
        try:
            macs, params = get_model_complexity_info(model, tuple(input_tensor.shape[1:]), as_strings=False,
                                                     print_per_layer_stat=True,
                                                     verbose=True)
        except:
            macs, params = get_model_complexity_info(model, tuple(input_tensor.shape), as_strings=False,
                                                     print_per_layer_stat=True,
                                                     verbose=True)
        # 打印模型摘要信息
        model_summary = summary(model, input_tensor)

        # 使用 thop 计算模型的 FLOPs 和参数量
        f, p = profile(model, inputs=[input_tensor])
        # f, p = clever_format([f, p], '%.3f')
        try:
            print(f"Complexity: {macs / 1e9} GMACS, Parameters: {params / 1e6} M")
            # print(f"FLOPs: {f}FLOPs, Parameters: {p}")
            print('FLOPs = ' + str(f / 1e9) + 'G')
            print('Params = ' + str(p / 1e6) + 'M')
        except:
            print('something wrong!')
        print(model_summary)
        # 输出模型的输出形状和数据类型
        output = model(input_tensor)
        print([get_shape_or_length(x) for x in output])

        # 预热 GPU，避免初次调用的额外开销（仅在使用 GPU 时需要）
        for _ in range(10):
            _ = model(input_tensor)

        # 开始计时
        start_time = time.time()

        # 运行推理
        with torch.no_grad():  # 禁用梯度计算以加速推理
            output = model(input_tensor)

        # 结束计时
        end_time = time.time()

        # 计算推理时间
        elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
        print(f"Inference time: {elapsed_time:.2f} ms")


def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config