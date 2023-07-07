import torch
from typing import Any

class DropFunction_Base:
    '''
    To change a drop model, simply override the 'drop_method' function.
    drop = NewDropFunction(params)
    X = drop(X, param1, param2, **kwargs)
    '''
    def __init__(self) -> None:                     pass
    
    def __call__(self, X, *args, **kwargs)->Any:    return self.drop_method(X, *args, **kwargs)
    
    def drop_method(self, X, *args, **kwargs)->Any: raise NotImplementedError
    
    def __repr__(self) -> str:
        s = ''
        for k, v in self.__dict__.items():
            s += f'{k}:{v} '
        return s

class DropSteadyRate(DropFunction_Base):
    def __init__(self, drop_rate=0.5, name='steady-droprate') -> None:
        super().__init__()
        self.name = name
        self.drop_rate = drop_rate

    def drop_method(self, X, *args, **kwargs):
        idx = torch.rand(X.shape) < self.drop_rate
        X[idx] = 0
        return X

class DropLinearChange(DropFunction_Base):
    def __init__(self, start=0.1, end=0.9, change_rate=0.005, name='linear-change-droprate') -> None:
        super().__init__()
        self.name = name
        self.start, self.end, self.change_rate = (start, end, change_rate)

    def drop_method(self, X, epoch, **kwargs):
        if(self.start < self.end): # increasing
            self.drop_rate = max(self.end, self.start + self.change_rate * epoch)
        else: # decreasing
            self.drop_rate = min(self.end, self.start - self.change_rate * epoch)
        idx = torch.rand(X.shape) < self.drop_rate
        X[idx] = 0
        return X

class DropExponentialDimish(DropFunction_Base):
    def __init__(self, start=0.9, end=0.1, diminish_rate=0.1, name='exponential-diminish-droprate') -> None:
        super().__init__()
        self.name = name
        self.start, self.end, self.diminish_rate = (start, end, diminish_rate)

    def drop_method(self, X, epoch, **kwargs):
        self.drop_rate = (self.start-self.end) * torch.exp(-self.diminish_rate * epoch) + self.end
        idx = torch.rand(X.shape) < self.drop_rate
        X[idx] = 0
        return X
