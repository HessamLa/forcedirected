from .algorithms import *

class embedder_model():
    def __init__(self, Z, mass, hops, alpha, std0=1, 
                        outputfilename='embedding.npy', logfilename='log.txt', *args, **kwargs):
        self.Z = Z
        self.mass = mass
        self.hops = hops
        self.alpha = alpha
        self.std0 = std0
        self.outputfilename = outputfilename
        self.logfilename = logfilename
        pass
    def run():
        pass
