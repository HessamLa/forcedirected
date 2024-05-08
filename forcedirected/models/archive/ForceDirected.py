from forcedirected.utilityclasses import Model_Base
import torch
from forcedirected.utilities import batchify
from recursivenamespace import rns

class ForceDirected(torch.nn.Module, Model_Base):
    """Force-Directed Base Model"""
    VER_MAJ="02"
    DESCRIPTION="Force-Directed Base Model"
    def __init__(self, *args, 
                lr:float = 1.0,
                verbosity:int = 2, # 0 no output, 2 short msg, 3:full msg, 4 short/full msg + exception msg, 6 exception msg + raise exception
                **kwargs) -> None:
        """
        lr: float, default=1.0, learning rate
        verbosity: int, default=2, verbosity level, 0:no output, 2:short msg, 3:full msg, 4:short/full msg + exception msg, 6:exception msg + raise exception
        """
        Model_Base.__init__(self, **kwargs) 
        torch.nn.Module.__init__(self)
        self.train = self.embed # alias

        self.verbosity = verbosity
        self.dZ = None
        self.lr = lr # should be defined in the derived class
        self.latest_epoch = 0 # latest epoch processed by the model

    def __str__(self) -> str:
        # return name of the class along with the version
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN} - {self.DESCRIPTION}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} v{self.VER_MAJ}.{self.VER_MIN}"
    
    def get_embeddings(self, detach=True):   
        """Returns embeddings as a numpy object"""
        return self.Z.detach().cpu().numpy()
    
    def get_embeddings_df(self, columns=None):
        """Returns embeddings as a pandas dataframe, with nodes as indices"""
        import pandas as pd
        import numpy as np
        node_labels = list(self.Gx.nodes())
        emb = self.get_embeddings()

        # Convert embeddings to a DataFrame
        # columns = ['node_label'] + [f'dim_{i+1}' for i in range(emb.shape[1])]
        data = np.column_stack([node_labels, emb])
        df = pd.DataFrame(data, columns=columns)
        return df
    
    def forward(self, bmask, **kwargs): 
        """forward pass, to calculate the forces and dZ
        example: 
        class FDModel(ForceDirected):
            ...
        F = FDModel()
        for epoch in range(1, epochs+1):
            ...
            F.dZ[bmask] = F(bmask, **kwargs)
            F.updateZ()
            ...
        """
        raise NotImplementedError("forward(.) is not implemented")
    
    def updateZ(self, lr=None):
        if(lr is None): lr = self.lr
        self.Z += self.dZ*lr

    @torch.no_grad()
    def embed(self, epochs=100, device='cpu', row_batch_size='auto', lr=None, Z=None, start_epoch=1, **kwargs):
        """
        Train the model to embed the graph.
        """
        # train begin
        if(start_epoch > epochs):
            raise ValueError(f"start_epoch should be <= epochs. start_epoch: {start_epoch}, epochs: {epochs}")
        kwargs = rns(kwargs)
        kwargs.epochs = epochs
        kwargs.start_epoch = start_epoch
        
        self.notify_train_begin_callbacks(**kwargs)

        # continue on an existing embedding
        if(Z is not None): 
            self.Z = Z 
        
        if(self.dZ is None):
            self.dZ = torch.nn.Parameter(
                        # torch.zeros_like(self.Z, device=device),
                        torch.zeros_like(self.Z),
                        requires_grad=False)            
        
        self.to(device)

        from forcedirected.utilities import optimize_batch_count
        @optimize_batch_count(max_batch_count=self.Z.shape[0])
        def run_batches(batch_count=1, **kwargs):
            kwargs = rns(kwargs)
            kwargs.batch_count = batch_count
            # print(f"run_batches: batch count: {kwargs['batch_count']}")
            n = self.Z.shape[0]
            batch_size = int(n/batch_count + 0.5) # ceiling of total/count
            for i, bmask in enumerate (batchify(list(range(n)), batch_size=batch_size)):
                # batch begin
                kwargs.batch = i+1
                kwargs.batch_size = batch_size
                self.notify_batch_begin_callbacks(**kwargs)
                if(self.verbosity >=3 and kwargs.batch_count > 1):
                    print(f"  batch {kwargs.batch}/{kwargs.batch_count}")
                ###################################
                # this is the forward pass
                self.dZ[bmask] = self.forward(bmask, **kwargs)
                # batch ends
                self.notify_batch_end_callbacks(**kwargs)    
            return batch_count, batch_size
        
        for epoch in range(start_epoch, epochs+1):
            if(self.stop_training): break
            
            # epoch begin
            # kwargs['epoch'] = epoch  
            kwargs.epoch = epoch
            self.notify_epoch_begin_callbacks(**kwargs)
            if(self.verbosity>=1): 
                print(f"Epoch {epoch}/{epochs}", end='')
            self.dZ.zero_() # fill self.dZ with zeros

            # kwargs['batch_count'], kwargs['batch_size'] = run_batches(**kwargs)
            kwargs.batch_count, kwargs.batch_size = run_batches(**kwargs)
            # get the average dZ
            dZ_norm_avg = torch.norm(self.dZ, dim=-1).mean().item()
            if(self.verbosity>=2):
                if(kwargs.batch_count > 1):
                    print(f"({kwargs.batch_count} batches)", end='')
                print(f"  dZ norm avg: {dZ_norm_avg:.4f}", end='')
                print("")
            self.updateZ(lr=lr)
            self.notify_epoch_end_callbacks(**kwargs)
            # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass
