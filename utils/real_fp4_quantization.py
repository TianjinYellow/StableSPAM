import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from triton.tools.experimental_descriptor import TmaDescKernelParam
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import pdb

from .kernel import *

# kernel 指南： MXFP4 Tensor 接受的data是x / scale之后的 Float32
# 我需要自己先计算scale存下来

def pack_scale(x, VEC_SIZE=16):
    M, K = x.shape
    x0 = x.abs()[:, ::VEC_SIZE]  
    x0 = x0.view(M // 128, 128, K // VEC_SIZE) 
    x0 = x0.view(M // 128, 32, 4, K // VEC_SIZE)  
    x0 = x0.view(M // 128, 32, 4, (K // VEC_SIZE) // 4, 4) 
    x_scale = x0.permute(0, 3, 1, 2, 4)
    return x_scale

def unpack_scale(packed, VEC_SIZE=16):
    temp = packed.permute(0, 3, 2, 1, 4) 
    temp = temp.reshape(packed.shape[0] * 128, packed.shape[1] * 4)
    return temp.repeat_interleave(VEC_SIZE, dim=1)


# 先写一个quantization函数，接受一个float32 tensor: x，返回q_x，scale
def fp4_quantization_step1(x : torch.Tensor,configs):
    x = x.clone()
    x_shape = x.shape
    x = x.view(-1,256)
    x_packed_shape = x.shape
    scale = x.abs().max(dim=-1).values.clamp(min = 1e-6)
    q_max = scale
    q_min = - q_max
    x_clamped = torch.clamp(x, q_min[:, None], q_max[:, None])
    x_bit = x_clamped / scale.unsqueeze(1)
    dummy = MXFP4Tensor(data = x_bit,device = x.device()) #这里data已经转换成uint8形式了
    packed = dummy.to_packed_tensor(dim = 1) # 一个uint8装两个fp4
    scale = pack_scale(scale.unsqueeze(1))
    return scale, packed, x_shape, x_packed_shape

def fp4_quantization_step2(x : torch.Tensor,configs):
    x = x.clone()
    x_shape = x.shape
    x = x.view(-1,256)
    x_packed_shape = x.shape
    scale = torch.ones_like(x,device = x.device())
    q_max  = scale.max(dim=-1).values
    q_min = - q_max
    x_clamped = torch.clamp(x, q_min[:, None], q_max[:, None])
    x_bit = x_clamped / scale
    dummy = MXFP4Tensor(data = x_bit,device = x.device()) #这里data已经转换成uint8形式了
    packed = dummy.to_packed_tensor(dim = 1) # 一个uint8装两个fp4
    scale = pack_scale(scale)
    return scale, packed, x_shape, x_packed_shape

def fp4_dequantization_step1(scale, packed, x_origin_shape, x_packed_shape):
    dummy = MXFP4Tensor()
    dummy.data = dummy.unpack_packed_tensor(packed_tensor = packed, dim = 1, original_shape = x_packed_shape )
    unpack = dummy.to(torch.float32)
    unpack.view(x_origin_shape)
    return unpack

def fp4_quantization_all(x: torch.Tensor, output_shape = None, transpose = False):
    scale, packed, x_shape, x_packed_shape = fp4_quantization_step1(x)
    unpack = fp4_dequantization_step1(scale, packed, x_shape, x_packed_shape)
    if transpose:
        unpack = unpack.T
    scale_1, packed_1, x_shape_1, x_packed_shape_1 = fp4_quantization_step1(unpack)
    return scale, packed_1,x_shape_1, x_packed_shape_1


def fp4_dequantization_all(scale, packed, x_packed_shape):
    dummy = MXFP4Tensor()
    dummy.data = dummy.unpack_packed_tensor(packed_tensor = packed, dim = 1, original_shape = x_packed_shape )
    unpack = dummy.to(torch.float32)
    scale = unpack_scale(scale)
    unpack = scale * unpack
    return unpack

class FP4Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, bias: Tensor,configs):
        # ---- debug
        x_scale, x_packed, x_shape, x_packed_shape = fp4_quantization_all(x)
        w_scale, w_packed, w_shape, w_packed_shape = fp4_quantization_all(w,transpose = True)
        ctx.save_for_backward(x_scale, x_packed, x_shape, x_packed_shape, w_scale, w_packed, w_shape, w_packed_shape,bias)

        x_scale = x_scale.to(torch.float8_e4m3fn)
        w_scale = w_scale.to(torch.float8_e4m3fn)

        x_desc = TmaDescKernelParam(x_packed.data_ptr(), x_shape, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE], 1)
        w_desc = TmaDescKernelParam(w_packed.data_ptr(), w_shape, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE], 1) 
    #     configs = {
    #     "BLOCK_SIZE_M": 128,
    #     "BLOCK_SIZE_N": 256,
    #     "BLOCK_SIZE_K": 256,
    #     "num_stages": 4,
    #     "ELEM_PER_BYTE": 2,
    #     "VEC_SIZE": 16,
    # }
        output = block_scaled_matmul(x_desc, x_scale, w_desc,w_scale, torch.float32,x_shape[0],w_shape[1],x_shape[1],configs)
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_scale, x_packed, x_shape, x_packed_shape, w_scale, w_packed, w_shape, w_packed_shape,bias = ctx.saved_tensors

        weight = fp4_dequantization_all(w_scale,w_packed,w_packed_shape)
        weight = weight.T

        quant_x = fp4_dequantization_all(x_scale,x_packed,x_packed_shape)

        grad_input =  grad_output @ weight

        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                                      quant_x.reshape(-1, quant_x.shape[-1])
        
        if bias is not None:
            out_features = bias.shape[0]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias,None



class real_fp4linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device = None,
        dtype = None,
        weight_data = None,
        bias_data = None,
        group_size = 256,
        stochastic_round = True,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.stochastic_round = stochastic_round
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.uint8, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter('bias', None)
        
        if weight_data is not None:
            self.weight.data.copy_(weight_data)
        if bias_data is not None and bias:
            self.bias.data.copy_(bias_data)

        self.configs = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 256,
            "num_stages": 4,
            "ELEM_PER_BYTE": 2,   # nvfp4
            "VEC_SIZE": 16,       # nvfp4
        }
    def forward(self, input:Tensor) -> Tensor:
        return FP4Linear.apply(input, self.weight, self.bias, self.configs)


def prepare_model_for_real_fp4_training_simulation_act_weight(model, args, target_module):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prepare_model_for_int8_training_simulation_act_weight(module, args, target_module)

        if isinstance(module, nn.Linear):
            if not name in target_module:
                print('Keep in original linear layer', name, module)
                continue
            
            # NOTE(hanqing): no need to pass those stuffs
            bias_data = module.bias.data if module.bias is not None else None
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            weight_data = module.weight.data
            new_layers = real_fp4linear(in_features, out_features, bias=bias, device='cuda:0', 
                weight_data=weight_data, bias_data=bias_data, 
                num_bits=args.weight_bits, group_size=args.weight_group_size, stochastic_round=args.stochastic_round)

            model._modules[name] = new_layers

    return model
