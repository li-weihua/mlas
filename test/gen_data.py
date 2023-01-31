import torch
import torch.nn as nn
import numpy as np

torch.set_grad_enabled(False)
torch.manual_seed(1)

in_channels = 48
out_channels = 48

conv = nn.Conv2d(in_channels, out_channels, (2, 3), padding=(0,1))
weight = conv.weight
bias = conv.bias

x = torch.randn(1, in_channels, 2, 40)
y = conv(x)

print(f'input: {x.min()}, {x.max()}')
print(f'output: {y.min()}, {y.max()}')

# save data
with open('./convdata/weight.data', 'wb') as f:
    f.write(weight.numpy().tobytes())

with open('./convdata/bias.data', 'wb') as f:
    f.write(bias.numpy().tobytes())

with open('./convdata/input.data', 'wb') as f:
    f.write(x.numpy().tobytes())

with open('./convdata/output.data', 'wb') as f:
    f.write(y.numpy().tobytes())
