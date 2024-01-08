# %%
import torch
import numpy as np
# %%
class Parent():
    def __init__(self, **kwargs) -> None:
        self._a=kwargs.get('a', 1)
        self._b=kwargs.get('b', 2)
        pass
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

class Child(torch.nn.Module, Parent):
    def __init__(self, **kwargs) -> None:
        Parent.__init__(self, **kwargs) 
        torch.nn.Module.__init__(self)
        _c=kwargs.get('c', 3)
        _d=kwargs.get('d', 4)
        # a random tensor with size 5x3x2
        z = torch.rand(4, 2, 3)
        
        self.register_buffer('_c', torch.tensor(_c))
        self.register_parameter('_d', torch.nn.Parameter(torch.tensor(_d), requires_grad=False))
        self.z = torch.nn.Parameter(z, requires_grad=False)
        self.dz = torch.nn.Parameter(torch.zeros_like(z), requires_grad=False)

    @property
    def c(self):
        return self._c
    @property
    def d(self):
        return self._d
    def __repr__(self):
        return super().__repr__()
    def forward(self, bmask, **kwargs):
        # bmask = torch.tensor(bmask)
        # bmask = torch.where(bmask)[0]

        self.z[bmask] -= 0.1
        
        
c = Child(a=10, b=20, c=30, d=40)
print(c.a, c.b, c.c)
print(c.d)
# TypeError: cannot assign 'torch.LongTensor' as parameter '_d' (torch.nn.Parameter or None expected)
print(c.d)
print(c.z)
print(c.forward(c.z>0.5))
print(c.z)

bmask = [3,2]
print(c.z[bmask])

# %%
