import torch

def fn(x):
    return torch.rand_like(torch.relu(x))

fn_s = torch.jit.script(fn)

x = torch.rand((2, 2), requires_grad=True)

fn_s(x)
fn_s(x)
fn_s(x)
fn_s(x)
