import torch.nn as nn

__all__ = ['Flatten']

class Flatten(nn.Module):
    #def __init__(self):
    #    super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size()[0],-1)
#class Linear_(nn.Module):
    #def __init__(self):
    #    super(Linear_, self).__init__()
#    def forward(self,x,output_size):
#        return nn.Linear(x.size()[1],output_size)
