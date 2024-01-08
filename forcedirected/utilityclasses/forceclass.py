from typing import Any

class ForceClass:
    def __init__(self, name, func, shape=None, dimension_coef=None, **kwargs) -> None:
        self.name = name
        self.func = func
        self.shape = shape
        self.dimension_coef = dimension_coef
        self.default_params = kwargs
        if(self.dimension_coef is not None):
            if(self.dimension_coef.shape!=self.shape):
                raise ValueError(f"dimension_coef shape is {self.dimension_coef.shape}. It must be {self.shape}")
        self.F = None
        pass

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __call__(self, *args: Any, suppress_default_params=False, **kwds: Any) -> Any:
        """
        fd_model is the forcedirected model which includes information about points embeddings.
        Upon calling the instantiated ForceClass_Base obj, the force is calculated and stored in fd_model.F
        """
        if(not suppress_default_params):
            kwds.update(self.default_params)
        self.F = self.forward(*args, **kwds)
        if(self.dimension_coef is not None):
            self.F = self.F * self.dimension_coef
        return self.F
    
class ForceClass_base():
    def __init__(self, name) -> None:
        self._name = name
    
    @property
    def name(self):
        return self._name
    
    def forward(self, *args, **kwargs):
        """force function"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
