from forcedirected.utilityclasses import Model_Base
import torch
from forcedirected.utilities import batchify

class ForceDirected(torch.nn.Module, Model_Base):
    """Force Directed Base Model"""
    VER_MAJ="02"
    DESCRIPTION="Force-Directed Base Model"
    def __init__(self, *args, **kwargs) -> None:
        """
        Gx is a Networkx graph object.
        n_dim is the number of dimensions to embed the graph.
        """
        Model_Base.__init__(self, **kwargs) 
        torch.nn.Module.__init__(self)
        # self.degrees = [d for n, d in Gx.degree()]
        
        self.dZ = None

    def __str__(self) -> str:
        # return name of the class along with the version
        return f"{self.__class__.__name__} v{self.VERSION} - {self.DESCRIPTION}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} v{self.VERSION}"
    
    def get_embeddings(self):   
        return self.Z.detach().cpu().numpy()
    
    def forward(self, bmask, **kwargs): 
        """forward pass, to calculate the forces and dZ
        example: 
        self.dZ[bmask] = sum([F(self, bmask, **kwargs) for F in self.Forces])
        """
        raise NotImplementedError("forward(.) is not implemented")
    
    def updateZ(self):
        self.Z += self.dZ*self.lr
        
    def train(self, epochs=100, device='cpu', row_batch_size='auto', **kwargs):
        self.embed(epochs=epochs, device=device, 
                   row_batch_size=row_batch_size, **kwargs)

    @torch.no_grad()
    def embed(self, epochs=100, device='cpu', row_batch_size='auto', lr=1.0, Z=None, **kwargs):
        # train begin
        kwargs['epochs'] = epochs
        self.notify_train_begin_callbacks(**kwargs)

        self.to(device)
        self.lr = lr # learning rate
        if(Z is not None): self.Z = Z # in case the model is to continue on an existing embedding
        
        # self.Z = self.Z.to(device)
        if(self.dZ is None):
            self.dZ = torch.nn.Parameter(
                        # torch.zeros_like(self.Z, device=device),
                        torch.zeros_like(self.Z),
                        requires_grad=False)            
        
        from forcedirected.utilities import optimize_batch_count
        @optimize_batch_count(max_batch_count=self.Z.shape[0])
        def run_batches(batch_count=1, **kwargs):
            kwargs['batch_count'] = batch_count
            # print(f"run_batches: batch count: {kwargs['batch_count']}")
            batch_size=int(self.Z.shape[0]/batch_count +0.5) # ceiling of total/count
            for i, bmask in enumerate (batchify(list(range(self.Z.shape[0])), batch_size=batch_size)):
                # batch begin
                kwargs['batch'] = i+1
                kwargs['batch_size'] = batch_size
                self.notify_batch_begin_callbacks(**kwargs)
                
                ###################################
                # this is the forward pass
                self.dZ[bmask] = self.forward(bmask, **kwargs)
                
                # batch ends
                self.notify_batch_end_callbacks(**kwargs)
            
            return batch_count, batch_size
        
        for epoch in range(epochs):
            if(self.stop_training): break
            
            # epoch begin
            kwargs['epoch'] = epoch
            self.notify_epoch_begin_callbacks(**kwargs)
            
            self.dZ.zero_() # fill self.dZ with zeros
            
            batch_count, batch_size = run_batches(**kwargs)
            kwargs['batch_count'], kwargs['batch_size'] = batch_count, batch_size

            self.updateZ()
        
            self.notify_epoch_end_callbacks(**kwargs)
            # epoch ends            
        
        self.notify_train_end_callbacks(**kwargs)
        pass