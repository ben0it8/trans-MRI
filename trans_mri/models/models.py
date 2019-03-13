import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .model_utils import *

def add_conv(c_in, c_out, ks, s=1, p=1, act_fn=nn.ELU, with_bn=False):
    conv = nn.Sequential(
        nn.Conv3d(c_in, c_out, ks, stride=s, padding=p), 
        act_fn())
    if with_bn: conv.add_module('2', nn.BatchNorm3d(c_out))
    return conv

def add_deconv(c_in, c_out, ks, s=1, p=1, act_fn=nn.ELU, with_bn=False):
    deconv = nn.Sequential(
        nn.ConvTranspose3d(c_in, c_out, ks, stride=s, padding=p), 
        act_fn())
    if with_bn: deconv.add_module('2', nn.BatchNorm3d(c_out))
    return deconv

def default_encoder(act_fn=nn.ELU, return_indices=True, with_bn=False):
    return nn.ModuleList([
            add_conv(1, 8, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxPool3d(2, stride=None, return_indices=return_indices),  
            add_conv(8, 16, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxPool3d(2, stride=None, return_indices=return_indices),
            add_conv(16, 32, 3, act_fn=act_fn, with_bn=with_bn),     
            nn.MaxPool3d(2, stride=None, return_indices=return_indices),
            add_conv(32, 16, 3, act_fn=act_fn, with_bn=with_bn, p=0)])
            
def default_decoder(act_fn=nn.ELU, act_fn_out=nn.Sigmoid, with_bn=False):
    act_fn_out = act_fn if act_fn_out is None else act_fn_out
    return nn.ModuleList([
            add_deconv(16, 32, 3, act_fn=act_fn, with_bn=with_bn, p=0),
            nn.MaxUnpool3d(2, stride=None),
            add_deconv(32, 16, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxUnpool3d(2, stride=None),
            add_deconv(16, 8, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxUnpool3d(2, stride=None),
            add_deconv(8, 1, 3, act_fn=act_fn_out, with_bn=with_bn)])
            # add_deconv(16, 1, 3, act_fn=act_fn_out, with_bn=with_bn)])

def default_classifier(in_features, num_classes, dropout1=0.25):
    return nn.ModuleList([nn.Linear(in_features, 32),
                          nn.ReLU(),
                          nn.Dropout(p=dropout1),
                          nn.Linear(32, num_classes)])

def encoder_classifier(act_fn=nn.ELU, return_indices=False, with_bn=False):
    return nn.ModuleList([
            add_conv(1, 8, 3, act_fn=act_fn, with_bn=with_bn),
            add_conv(8, 8, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxPool3d(2, stride=2, return_indices=return_indices),  
            add_conv(8, 16, 3, act_fn=act_fn, with_bn=with_bn),
            add_conv(16, 16, 3, act_fn=act_fn, with_bn=with_bn),
            nn.MaxPool3d(2, stride=2, return_indices=return_indices),
            add_conv(16, 32, 3, act_fn=act_fn, with_bn=with_bn),   
            add_conv(32, 32, 3, act_fn=act_fn, with_bn=with_bn),     
            nn.MaxPool3d(2, stride=2, return_indices=return_indices),
            add_conv(32, 64, 3, act_fn=act_fn, with_bn=with_bn),     
            add_conv(64, 64, 3, act_fn=act_fn, with_bn=with_bn), 
            add_conv(64, 64, 3, act_fn=act_fn, with_bn=with_bn),     
            nn.MaxPool3d(3, stride=3, return_indices=return_indices)])        

class ConvClassifier(nn.Module):
    def __init__(self, input_shape:tuple, act_fn=nn.ELU, with_bn=False, dropout=0.2, num_classes=2, debug=False):
        super().__init__()   
        self.input_shape=list(input_shape)
        self.debug=debug
        self.encoder = default_encoder(act_fn=act_fn, return_indices=False)

        num_features = self.get_num_ftrs()
        
        self.classifier = nn.Sequential(nn.Linear(num_features, 64),
                                         act_fn(),
                                         nn.Dropout(dropout),
                                         nn.Linear(64, num_classes))
        
        self.apply(init_weights)
        if self.debug:
            print("[Debug mode ON]")
            out = self.forward(Variable(torch.ones(1,1,*self.input_shape)))
        print(f"ConvClassifier intiailized (no. trainable params: {get_num_params(self)})")

    def flatten(self, x):
        return x.view(x.size(0), -1)
                                                   
    def get_num_ftrs(self):
        inp = Variable(torch.ones(1,1,*self.input_shape))
        f = self.encode(inp)
        return int(np.prod(f.size()[1:]))
                                                   
    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x

    def classify(self,x):
        return self.classifier(x)
    
    def forward(self,x):
        x = self.encode(x)
        x = self.flatten(x)
        x = self.classify(x)
        return x
                                                                                  
class ConvAutoEncoder(nn.Module):
    
    def __init__(self, input_shape:tuple, act_fn=nn.ELU, act_fn_out=nn.Sigmoid, 
                 noise:float=0.0, with_classifier=False, num_classes:int=2, debug:bool=False):
        super().__init__()   
        self.input_shape=list(input_shape)
        self.debug=debug
        self.encoder = default_encoder(act_fn=act_fn)                                 
        self.decoder = default_decoder(act_fn=act_fn, act_fn_out=act_fn_out)
        self.noise=noise
        

        if with_classifier:
            latent_dim = get_latent_dim(self, self.input_shape)
            self.classifier = nn.Linear(latent_dim, num_classes)
            print("NOTE: CAE initialized with additional `classifier` module.")

        self.apply(init_weights)
        if self.debug:
            print("[Debug mode ON]")
            out = self.forward(Variable(torch.ones(1,1,*self.input_shape)))
        
        print(f"ConvAutoEncoder intiailized (no. trainable params: {get_num_params(self)})")
        print(f"input noise: {self.noise}")

    def encode(self, x, pool_dict={"inds": [], "shapes": []}, 
               return_dict = False):
        if self.debug: print("encoder:")
        for layer in self.encoder:
            if isinstance(layer, nn.modules.MaxPool3d):
                x, idx = layer(x)
                pool_dict["inds"].append(idx)
                pool_dict["shapes"].append(x.size())
            else: x = layer(x)
            if self.debug: print(x.shape)
        
                
        if return_dict: return x, pool_dict
        else: return x
        
    def decode(self, x, pool_dict):
        if self.debug: print("decoder:")

        for layer in self.decoder:
            if isinstance(layer, nn.modules.MaxUnpool3d):
                x = layer(x, pool_dict["inds"].pop(), 
                          output_size=pool_dict["shapes"].pop())
            else:
                x = layer(x)
            if self.debug: print(x.shape)

        return x
    
    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, x):
        if self.noise > 0.0: x = F.dropout3d(x, p=self.noise)

        pool_dict = {"inds": [], "shapes": [x.size()]}
        
        encoded, pool_dict = self.encode(x, pool_dict, return_dict=True)
                
        if self.debug: print(encoded.flatten().size())
        if hasattr(self,"classifier"): 
            logits = self.classifier(F.dropout(self.flatten(encoded), p = 0.35, training=self.training))
        pool_dict["shapes"].pop()
        
        decoded = self.decode(encoded, pool_dict)
        
        if hasattr(self, "classifier"): return decoded, logits
        else: return decoded         
    
class EncoderBlock(nn.Module):
    
    def __init__(self, c_in, c_out, c_out_str, dropout=0.1, act_fn=nn.ELU):
        super().__init__()
        self.p=dropout
        self.conv = nn.Sequential(
                nn.Conv3d(c_in, c_out, 3, stride=1, padding=1), 
                act_fn())
        
        self.conv_stride = nn.Sequential(
                nn.Conv3d(c_in+c_out, c_out_str, 4, stride=2, padding=0), 
                act_fn())
          
    def forward(self, x):
        h = F.dropout3d(self.conv(x), p=self.p, training=self.training) if self.p>0.0 else self.conv(x) 
        z = torch.cat([h, x], 1) 
        return self.conv_stride(z)
    
class DecoderBlock(nn.Module):
    
    def __init__(self, c_in, c_out, c_out_str, act_fn=nn.ELU, act_fn_out=nn.Sigmoid, output_padding=0):
        super().__init__()
        act_fn_out = act_fn if act_fn_out is None else act_fn_out
        
        self.conv = nn.Sequential(
                nn.Conv3d(c_in, c_out, 3, stride=1, padding=1),
                act_fn())
        
        self.conv_trans = nn.Sequential(
                nn.ConvTranspose3d(c_in+c_out, c_out_str, 4, stride=2, padding=0, output_padding=output_padding),
                act_fn_out())
    
    def forward(self, x):
        h = self.conv(x)
        z = torch.cat([h, x], 1)
        return self.conv_trans(z)

class StridedAutoEncoder(nn.Module):
    
    def __init__(self, input_shape, act_fn=nn.ELU, act_fn_out=None, noise=0.0, 
                 with_classifier=False, num_classes=2, debug=False):
        super().__init__()
        self.input_shape = list(input_shape)
        self.debug=debug
        self.noise=noise
        
        self.encoder=nn.ModuleList([
                    EncoderBlock(1, 7, 8, dropout=0.1, act_fn=act_fn),
                    EncoderBlock(8, 7, 16, dropout=0.1, act_fn=act_fn),
                    EncoderBlock(16, 7, 16, dropout=0.1, act_fn=act_fn)])
        
        self.decoder=nn.ModuleList([
                    DecoderBlock(16, 7, 16, act_fn=act_fn, output_padding=(1,0,1)),
                    DecoderBlock(16, 7, 8, act_fn=act_fn, output_padding=(0,0,0)),
                    DecoderBlock(8, 7, 1, act_fn=act_fn, act_fn_out=act_fn_out)])
        
        if with_classifier:
            latent_dim = get_latent_dim(self, self.input_shape)
            self.classifier = nn.Linear(latent_dim, num_classes)
            print("NOTE: SCAE initialized with additional `classifier` module.")
        
        self.apply(init_weights)
        if self.debug:
            print("[Debug mode ON]")
            out = self.forward(Variable(torch.ones(1,1,*self.input_shape)))
        print(f"StridedConvAutoEncoder initialized. (no. trainable params: {get_num_params(self)})")
            
    def flatten(self, x):
        return x.view(x.size(0), -1)

    def encode(self, x):
        if self.debug: print('encoder:')
        for layer in self.encoder:
            x = layer(x)
            if self.debug: print(x.shape)
        return x
    
    def decode(self, x):
        if self.debug: print('decoder:')
        for layer in self.decoder:
            x = layer(x)
            if self.debug: print(x.shape)
        return x
    
    def forward(self, x):
        if self.debug: print('input:',x.shape)
        
        if self.noise > 0.0: 
            x = F.dropout3d(x, p=self.noise, training=self.training)
            
        x_enc = self.encode(x)
        if self.debug: print('code:', x_enc.flatten().size())
        if hasattr(self,"classifier"): 
            logits = self.classifier(F.dropout(self.flatten(x_enc), p = 0.5, training=self.training))

        x_dec = self.decode(x_enc)
        
        if x.shape != x_dec.shape:
            x_dec = F.interpolate(x_dec, size=self.input_shape, mode='nearest')
        
        if self.debug: print('output:',x_dec.shape)
        if hasattr(self, "classifier"): return x_dec, logits
        else: return x_dec         

class EncoderClassifier(nn.Module):


    """Module be used on top of a (pretrained) encoder coming from an AutoEncoder."""

    def __init__(self, input_shape:tuple, encoder:nn.ModuleList=None, num_classes:int=2,
                 dropout1=0.25):
        
        super().__init__()   
        self.input_shape = list(input_shape)

        if encoder is not None: 
            if isinstance(encoder, nn.modules.container.ModuleList):
                self.encoder = encoder
            elif isinstance(encoder, dict):
                self.encoder = default_encoder(return_indices=False)
                self.encoder.load_state_dict(encoder)
        else:
            self.encoder = default_encoder(return_indices=False)

        num_features = self.get_num_ftrs()
        
        self.classifier = default_classifier(num_features, num_classes, dropout1=dropout1)
        self.classifier.apply(init_weights)
        print(f"EncoderClassifier intiailized (no. trainable params: {get_num_params(self)})")
         
    def encode(self, x):
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool3d):
                x = layer(x)
                #pool returns indices too..
                if isinstance(x, tuple) and len(x) == 2: x=x[0] 
            else:
                x=layer(x)
        return x
    
    def classify(self, x):
        for layer in self.classifier:
            x = layer(x)
        return x
    
    def get_num_ftrs(self):
        inp = Variable(torch.ones(1,1,*self.input_shape))
        f = self.encode(inp)
        return int(np.prod(f.size()[1:]))

    def forward(self, x):
        x = self.encode(x)
        x = x.view(x.size(0),-1)
        x = self.classify(x)
        return x