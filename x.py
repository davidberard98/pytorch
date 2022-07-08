import torch

def fn(x, y):
    return torch.add(x, y)

print(torch.jit.script(fn).graph)
