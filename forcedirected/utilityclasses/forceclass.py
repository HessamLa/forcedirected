from typing import Any

class ForceClass:
    def __init__(self, name, func, shape=None, dimension_coef=None) -> None:
        self.name = name
        self.func = func
        self.shape = shape
        self.dimension_coef = dimension_coef
        if(self.dimension_coef is not None):
            if(self.dimension_coef.shape!=self.shape):
                raise ValueError(f"dimension_coef shape is {self.dimension_coef.shape}. It must be {self.shape}")
        self.F = None
        pass
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        fd_model is the forcedirected model which includes information about points embeddings.
        Upon calling the instantiated ForceClass_Base obj, the force is calculated and stored in fd_model.F
        """
        self.F = self.func(*args, **kwds)
        if(self.dimension_coef is not None):
            self.F = self.F * self.dimension_coef
        return self.F