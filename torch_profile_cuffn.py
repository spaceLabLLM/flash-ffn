import torch
from cuffn_module import FFN
from torch.profiler import profile, record_function, ProfilerActivity


class MLP(torch.nn.Module):

    def __init__(self, input_dim, middle_dim, bias=True, dtype=torch.float16):
        super().__init__()
        self.c_fc    = torch.nn.Linear(input_dim, middle_dim, bias=bias)
        # self.gelu    = torch.nn.GELU()
        self.c_proj  = torch.nn.Linear(middle_dim, input_dim, bias=bias)

    def forward(self, x):
        y = self.c_fc(x)
        # x = self.gelu(x)
        z = self.c_proj(y)
        return z


batch_size: int = 1
total_tokens: int = 800
input_dim: int = 1000
expert_cnt: int = 8
expert_dim: int = 800
tp_degree: int = 8

th_device = torch.device("cuda:0")
th_stream = torch.cuda.current_stream(th_device)

tokens_per_expert = total_tokens // expert_cnt
tp_expert_dim = expert_dim // tp_degree
ffn_model = FFN(input_dim, tp_expert_dim, False, th_device, torch.float16)

ref_model = MLP(input_dim, tp_expert_dim, False, torch.float16)
ref_model.half()
ref_model.to(th_device)

# ref_model.c_fc.weight.data.copy_(ffn_model.linear1_weight.data)
# ref_model.c_proj.weight.data.copy_(ffn_model.linear2_weight.data)
ref_model.c_fc.weight.data = ffn_model.linear1_weight.data
ref_model.c_proj.weight.data = ffn_model.linear2_weight.data
try:
    x = torch.randn([tokens_per_expert, input_dim], dtype=torch.float16, device=th_device)
    y = ffn_model.forward(x)
    y_ref = ref_model.forward(x)
    
    torch.cuda.synchronize()
    torch.testing.assert_close(y_ref, y, atol=0.1, rtol=1e-1)
except AssertionError:
    import traceback; traceback.print_exc()

with torch.profiler.profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
) as prof:
    with record_function("torch_linear"):
        x = torch.randn([tokens_per_expert, input_dim], dtype=torch.float16, device=th_device)
        torch.cuda.synchronize()
        y = ffn_model.forward(x)
        torch.cuda.synchronize()

# for evt in prof.key_averages():
#     print(evt.key)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))