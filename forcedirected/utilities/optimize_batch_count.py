# %%
import sys
import functools
from typing import Any
class optimize_batch_count:
    def __init__(self, *args, verbose=False, 
                min_batch_count=1, max_batch_count=sys.maxsize, 
                max_retries=3, out_of_memory_messages=[],
                **kwargs) -> None:
        if(verbose==False):
            self.vprint = lambda *args, **kwargs: None
        else:
            print("optimize_batch_count Verbose mode is on.")
            self.vprint = print

        self.out_of_memory_messages = ['out of memory', 'cannot allocate memory'] + out_of_memory_messages

        self.max_retries = max_retries
        self.retry_count = 0
        # default min and max values set through the decorator
        self.default_min_batch_count = min_batch_count 
        self.default_max_batch_count = max_batch_count 
        # min and max values set through the function call
        self.min_batch_count = None # last unsuccessful batch count +1
        self.max_batch_count = None # maximum allowed batch count
        self.batch_count = None # current batch count to be determined
        self.upper_batch_count = None # last successful batch count

    def __call__(self, func, *args:Any, **kwargs: Any) -> Any:
        @functools.wraps(func)
        def wrapper(*args, min_batch_count=self.default_min_batch_count, max_batch_count=self.default_max_batch_count,
                    **kwargs):
            # The search range is [[min_batch_count, upper_batch_count]]
            if(self.min_batch_count is None): # first call
                self.min_batch_count = min_batch_count
            if(self.max_batch_count is None): # first call
                self.max_batch_count = max_batch_count

            if(self.batch_count is None): # first call
                self.batch_count = kwargs.get('batch_count', self.min_batch_count)
            
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
                    self.retry_count = 0 # reset retry count
                    # print(f"last successful batch count: {self.upper_batch_count}")
                    return result
      
                except Exception as e:
                    is_out_of_memory = False
                    # read the exception message
                    for msg in self.out_of_memory_messages:
                        if msg in str(e).lower():
                            is_out_of_memory = True
                            break
                    # not an out of memory error
                    if(not is_out_of_memory): 
                        print(f"Exception: {e}")
                        raise e
                    # process out of memory exception
                    # Use binary search strategy to find the optimal batch count
                    if(self.batch_count > self.max_batch_count):
                        raise Exception("Maximum valid batch count is surpassed.")
                    if(self.retry_count >= self.max_retries):
                        raise Exception("Maximum number of retries for batch count optimization is reached.")                        
                    if(self.upper_batch_count is None): # a successful batch size has not yet been found
                        self.min_batch_count = self.batch_count+1
                        self.batch_count = min(self.batch_count*2, self.max_batch_count)
                    elif(self.upper_batch_count == self.min_batch_count): # the convergence point is not valid anymore, reset the upper bound
                        self.retry_count += 1
                        self.min_batch_count += 1
                        self.batch_count = self.min_batch_count
                        self.upper_batch_count *= 2
                    else:
                        self.min_batch_count = self.batch_count+1
                        self.batch_count = (self.min_batch_count + self.upper_batch_count)//2

                    if(self.min_batch_count > self.max_batch_count):
                        raise Exception(f"Max valid batch count is {self.max_batch_count}. No valid batch count was found.")
                        

        return wrapper

if __name__=='__main__':
    # @optimize_batch_count(max_batch_count=100)
    @optimize_batch_count(verbose=True)
    def func1(tag, batch_count=1, *args, **kwargs):
        """func1 docstring"""
        # batch_size = int(max_size/batch_count+0.5)
        if(batch_count<78):
            # print(f"func1 OOM")
            raise MemoryError("Out of memory")
        print(f"func1{tag} run. batch_count={batch_count}")
    
    @optimize_batch_count(out_of_memory_messages=['memory problem'], verbose=True)
    def func2(tag='', batch_count=1, *args, **kwargs):
        """func2 docstring"""
        if(batch_count<1026 and tag<10):
            # print(f"func1 OOM")
            raise RuntimeError("(Memory Problem!)")
        elif(batch_count<1800 and tag >=10):
            raise RuntimeError("(Memory Problem!)")
        print(f"func2{tag} run batch_count={batch_count}")
    
    print("starting iteration")
    for _ in range(20):
        # print("\nIteration", _+1)
        # func1(max_batch_count=100)
        # func1(batch_count=100, min_batch_count=50)
        #   func1(tag='-this-tag')
        #   func1(tag='-this-tag',)
        # func1()
        func2(tag=_, min_batch_count=10)
    
    # verify correct wrapping
    print(func1.__doc__)  
    print(func2.__doc__)
# %%
