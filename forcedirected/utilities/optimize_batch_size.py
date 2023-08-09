import sys
import functools
class optimize_batch_size:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        functools.update_wrapper(self, func)

        self.batch_size = sys.maxsize
        self.max_batch_size = sys.maxsize
        self.min_batch_size = None

    def __call__(self, *args, **kwargs):
        if(self.batch_size is None): # first call
            self.batch_size = kwargs.get('batch_size', self.batch_size)
        if(self.max_batch_size is None): # first call
            self.max_batch_size = self.batch_size
        if self.batch_size<self.max_batch_size: # batch_size was the last successful batch size
            self.min_batch_size = self.batch_size
            # upgrade batch_size
            self.batch_size = (self.batch_size + self.max_batch_size + 1)//2
            # print(f"Test new batch size ({min_batch_size},{batch_size},{max_batch_size})", file=sys.stderr)
        else:                        # either batch_size was the last unsuccessful batch size or it is to be determined
            self.min_batch_size = None
        
        def wrapper(self, *args, **kwargs):
            while True:
                try:
                    kwargs['batch_size'] = self.batch_size
                    kwargs['max_batch_size'] = self.max_batch_size
                    kwargs['min_batch_size'] = self.min_batch_size
                    return self.func(self, *args, **kwargs)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        if(self.batch_size == 1):
                            print("One row is too big for GPU memory. Please reduce the number of dimensions or the number of nodes.", file=sys.stderr)
                            raise e
                        if(self.min_batch_size is None): # a successful batch size has not been found yet
                            self.max_batch_size = self.batch_size-1
                            self.batch_size = self.max_batch_size//2
                        else:           # a successful batch size is in range [[min_batch_size, batch_size))
                            self.max_batch_size = self.batch_size-1
                            self.batch_size = (self.min_batch_size + self.max_batch_size)//2
                        # print(f"REDUCE batch_size: {self.batch_size}, max_batch_size: {self.max_batch_size}")
                        
                    else:
                        print(f"Exception: {e}")
                        raise e
        return wrapper(self, *args, **kwargs)