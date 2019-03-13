import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def get_num_params(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum([np.prod(p.size()) for p in trainable_params])
    return num_trainable

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            m.reset_parameters()
        elif isinstance(m, nn.Linear):
            m.reset_parameters()
            
def get_latent_dim(model, input_shape):
    dbg = model.debug; model.debug=False
    f = model.encode(Variable(torch.ones(1,1,*input_shape)))
    model.debug=dbg
    return int(np.prod(f.size()[1:]))

def check_trainable(model, module:str):
    module = getattr(model, module)
    is_trainable =  all([p.requires_grad for p in module.parameters()])
    num_trainable = get_num_params(module)
    return is_trainable, num_trainable