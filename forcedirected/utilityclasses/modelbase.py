from ..Functions import generate_random_points

class Callback_Base:
    def __init__(self, **kwargs) -> None:                   pass
    def on_epoch_begin(self, fd_model, epoch, **kwargs):    pass
    def on_epoch_end(self, fd_model, epoch, **kwargs):      pass
    def on_batch_begin(self, fd_model, batch, **kwargs):    pass
    def on_batch_end(self, fd_model, batch, **kwargs):      pass
    def on_train_begin(self, fd_model, epochs, **kwargs):   pass
    def on_train_end(self, fd_model, epochs, **kwargs):     pass

class Model_Base:
    def __init__(self, **kwargs) -> None:
        self.callbacks = []
        self.stop_training = False
        if('callbacks' in kwargs):
            self.add_callbacks(kwargs['callbacks'])
            
    def add_callbacks(self, callbacks):
        self.callbacks += callbacks
        
    ## EPOCH
    def notify_epoch_begin_callbacks(self, epoch, **kwargs):
        kwargs['epoch'] = epoch
        for callback in self.callbacks:
            callback.on_epoch_begin(self, **kwargs)
        pass
    def notify_epoch_end_callbacks(self, epoch, **kwargs):
        kwargs['epoch'] = epoch
        for callback in self.callbacks:
            callback.on_epoch_end(self, **kwargs)
        pass
    
    ## BATCH
    def notify_batch_begin_callbacks(self, batch, **kwargs):
        kwargs['batch'] = batch
        for callback in self.callbacks:
            callback.on_batch_begin(self, **kwargs)
        pass

    def notify_batch_end_callbacks(self, batch, **kwargs):
        kwargs['batch'] = batch
        for callback in self.callbacks:
            callback.on_batch_end(self, **kwargs)
        pass
    
    ## TRAIN
    def notify_train_begin_callbacks(self, epochs, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(self, epochs, **kwargs)
        pass
    def notify_train_end_callbacks(self, epochs, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(self, epochs, **kwargs)
        pass

class ForceDirected(Model_Base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)