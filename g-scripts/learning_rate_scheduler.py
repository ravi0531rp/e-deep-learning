from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np

class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002,decay=0.00002,warmup_epochs=2):
        self.num_passed_batchs = 0   
        self.warmup_epochs=warmup_epochs  
        self.lr=lr_base 
        self.decay=decay  
        self.steps_per_epoch=0 
    def on_batch_begin(self, batch, logs=None):
        if self.steps_per_epoch==0:
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1
    def on_epoch_begin(self,epoch,logs=None):
        print("learning_rate:",K.get_value(self.model.optimizer.lr)) 