# %%
import sys
import functools
from typing import Any
class optimize_batch_count:
    def __init__(self, func=None, verbose=False, max_batch_count=None, *args, **kwargs) -> None:
        if(verbose==False):
            self.vprint = lambda *args, **kwargs: None
        else:
            print("Verbose mode is on.")
            self.vprint = print

        self.func = func
        

        self.batch_count = None # current batch count
        self.upper_batch_count = None # last successful batch count
        self.min_batch_count = None # last unsuccessful batch count +1
        self.max_batch_count = None # maximum allowed batch count

    def __call__(self, func, min_batch_count=1, max_batch_count=sys.maxsize, *args:Any, **kwargs: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            # The search range is [[min_batch_count, upper_batch_count]]
            if(self.batch_count is None): # first call
              self.batch_count = kwargs.get('batch_count', 1)
            if(self.min_batch_count is None):
              self.min_batch_count = min_batch_count
            self.max_batch_count = max_batch_count
            
            # Set the min and and upper (if available) boundaries
            if (self.upper_batch_count is not None): # update batch_count
                self.batch_count = (self.min_batch_count + self.upper_batch_count)//2
            # Start the search
            while True:
                try:
                    self.vprint(f"@optimize_batch_count call '{func.__name__}' with batch_count={self.batch_count} (min:{self.min_batch_count}, upper:{self.upper_batch_count})")
                    kwargs['batch_count'] = self.batch_count
                    result = func(*args, **kwargs)
                    self.upper_batch_count = self.batch_count
                    # print(f"last successful batch count: {self.upper_batch_count}")
                    return result
      
                except Exception as e:
                    if 'out of memory' in str(e).lower():
                        # Use binary search strategy to find the optimal batch count
                        if(self.batch_count > self.max_batch_count):
                            raise Exception("Maximum valid batch count is surpassed.")
                        if(self.upper_batch_count is None): # a successful batch size has not yet been found
                            self.min_batch_count = self.batch_count+1
                            self.batch_count = min(self.batch_count*2, self.max_batch_count)
                        else:
                            self.min_batch_count = self.batch_count+1
                            self.batch_count = (self.min_batch_count + self.upper_batch_count)//2
                        if(self.min_batch_count > self.max_batch_count):
                            raise Exception("No valid batch count was found.")
                    else:
                        print(f"Exception: {e}")
                        raise e

        return wrapper

if __name__=='__main__':
    @optimize_batch_count(max_batch_count=60)
    def func1(tag, batch_count=1, *args, **kwargs):
        """func1 docstring"""
        # batch_size = int(max_size/batch_count+0.5)
        if(batch_count<78):
            # print(f"func1 OOM")
            raise MemoryError("Out of memory")
        print(f"func1{tag} run. batch_count={batch_count}")
    
    @optimize_batch_count()
    def func2(tag='', batch_count=1, *args, **kwargs):
        """func2 docstring"""
        if(batch_count<1026):
            # print(f"func1 OOM")
            raise RuntimeError("Out of memory")
        print(f"func2{tag} run batch_count={batch_count}")
    
    for _ in range(10):
      # print("\nIteration", _+1)
      # func1(max_batch_count=100)
      # func1(batch_count=100, min_batch_count=50)
      func1(tag='-this-tag')
      # func1()
      # func2(tag='-tag2', min_batch_count=10)
    
    # verify correct wrapping
    print(func1.__doc__)  
    print(func2.__doc__)
# %%
