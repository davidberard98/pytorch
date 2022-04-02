from typing import List
import torch

#torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

def fn(weight1, bias1, weight2, bias2, tensor_list: List[torch.Tensor]):
    catted = torch.cat(tensor_list, 1)
    '''
    split_sizes = [0] * sz
    for i in range(sz):
        x = tensor_list[i]
        xsz = x.size()
        r = xsz[1]
        split_sizes[i] = r
    '''
    linear = torch.nn.functional.linear(catted, weight1, bias1)
    result = torch.relu(linear)
    linear_1 = torch.nn.functional.linear(result, weight2, bias2)
    sigmoid = torch.sigmoid(linear_1)
    return sigmoid  #, split_sizes

def reduce_fn(x):
    return x.sum()

def get_inputs(a, b):
    weight1 = torch.rand((64+b, 384+2*a), requires_grad=True)
    bias1 = torch.rand((64+b), requires_grad=True)
    weight2 = torch.rand((1, 64+b), requires_grad=True)
    bias2 = torch.rand((1), requires_grad=True)

    src1 = torch.rand((1+a%2, 128+a), requires_grad=False)
    src2 = torch.rand((1+a%2, 256+a), requires_grad=False)

    return weight1, bias1, weight2, bias2, src1, src2

torch._C._jit_set_num_profiled_runs(2)
torch._C._jit_set_fusion_strategy([('STATIC', 4)])

sigmoid_target = torch.rand((4, 4))
fn_s = torch.jit.script(fn)
print("~~1")
print(fn_s.graph)
print("~~2")

for _ in range(20):
    a, b, c, d, e, f = get_inputs(_, _)
    for r in range(2):
        ret = fn_s(a, b, c, d, [e, f])
        reduce_fn(ret).backward()
a, b, c, d, e, f = get_inputs(0, 0)
torch._C._jit_make_first_backward_input_undefined(True)
ret = fn_s(a, b, c, d, [e[0:0, :], f[0:0, :]])
reduce_fn(ret).backward()
