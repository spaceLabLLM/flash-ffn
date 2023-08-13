import torch

import math
from cuffn import run_fnn_forward

class ffnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, linear1_weight, linear2_weight, linear1_bias, linear2_bias):
        # ctx.save_for_backward(x, linear1_weight, linear2_weight, linear1_bias, linear2_bias)
        # print("Shapes:")
        # print(x.shape, linear1_weight.shape, linear2_weight.shape)
        y = run_fnn_forward(x, linear1_weight, linear2_weight, linear1_bias, linear2_bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # x, linear1_weight, linear2_weight, linear1_bias, linear2_bias = ctx.saved_tensors
        # grad_x, grad_linear1_weight, grad_linear2_weight, grad_linear1_bias, grad_linear2_bias = run_fnn_backward(
        #     grad_output, x, linear1_weight, linear2_weight, linear1_bias, linear2_bias
        # )
        # return grad_x, grad_linear1_weight, grad_linear2_weight, grad_linear1_bias, grad_linear2_bias
        return grad_output

def ffn_func(x, linear1_weight, linear2_weight, linear1_bias, linear2_bias):
    return ffnFunc.apply(x, linear1_weight, linear2_weight, linear1_bias, linear2_bias)

class FFN(torch.nn.Module):
    __constants__ = ['in_features', 'proj_features']
    in_features: int
    proj_features: int
    linear1_weight: torch.Tensor
    linear2_weight: torch.Tensor
    linear1_bias: torch.Tensor
    linear2_bias: torch.Tensor
    
    def __init__(self, in_features: int, proj_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert bias is False
        super(FFN, self).__init__()
        self.in_features = in_features 
        self.proj_features = proj_features
        self.gelu = torch.nn.GELU()
        self.linear1_weight = torch.nn.Parameter(torch.empty((proj_features, in_features), **factory_kwargs))  # 
        self.linear2_weight = torch.nn.Parameter(torch.empty((in_features, proj_features), **factory_kwargs)) # 
        
        if bias:
            self.linear1_bias = torch.nn.Parameter(torch.empty(proj_features, **factory_kwargs)) #
            self.linear2_bias = torch.nn.Parameter(torch.empty(in_features, **factory_kwargs)) #
        else:
            self.register_parameter("linear1_bias", None)
            self.register_parameter('linear2_bias', None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.linear1_weight)
        torch.nn.init.uniform_(self.linear2_weight)
        
        if self.linear1_bias is not None:
            torch.nn.init.uniform_(self.linear1_bias)
        if self.linear2_bias is not None:  
            torch.nn.init.uniform_(self.linear2_bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = ffn_func(input, self.linear1_weight, self.linear2_weight, self.linear1_bias, self.linear2_bias)
        return output