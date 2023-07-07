from typing import Any

class ForceClass:
    def __init__(self, name, func) -> None:
        self.name = name
        self.func = func
        self.F = None
        pass
    def __call__(self, fd_model, *args: Any, **kwds: Any) -> Any:
        """
        fd_model is the forcedirected model which includes information about points embeddings.
        Upon calling the instantiated ForceClass_Base obj, the force is calculated and stored in fd_model.F
        """
        self.F = self.func(fd_model, *args, **kwds)
        return self.F