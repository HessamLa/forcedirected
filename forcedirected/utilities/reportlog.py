# %%
import pathlib, os
from typing import Any
from pprint import pprint, pformat

class ReportLog:
    def __init__(self, filepath=None) -> None:
        if(filepath is None):
            filepath = __file__
        p = pathlib.Path(filepath)
        p.parents[0].mkdir(parents=True, exist_ok=True)        
        
        self._path = p.absolute()
        try:
            os.remove(self._path)
        except:
            pass
        pass
    @property
    def path(self) -> str:
      return self._path

    def _make_str (self, *args: Any, **kwds: Any) -> str:
        string = ''
        for s in args:
            string += str(s)+' '
        if(len(kwds)>0):
          string += str(kwds)
        
        if(len(string)>0):
            string = string[:-1]
        return string
    
    def _tofile (self, string):
        with open(self._path, 'a') as f:
            f.write(string+'\n')

    def __call__(self, *args: Any, **kwds: Any)-> None:
        string = self._make_str(*args, **kwds)
        self._tofile(string)

    def print(self, *args: Any, **kwds: Any) -> None:
        string = self._make_str(*args, **kwds)
        self._tofile(string)
        print(string)

    def pprint(self, *args: Any) -> None:
        string = pformat(*args)
        # string = self._make_str(*args, **kwds)
        self._tofile(string)
        print(string)

    
        
if __name__=='__main__':
  logpath = __file__+'.log'
  log = ReportLog(logpath)
  log.print("This is passed to the file and the prompt", ['some', 'random'], ('thing'))
  log("This is ONLY passed to the file", ['some', 'random'], ('object'))
  print('Now check the file', log.path)
