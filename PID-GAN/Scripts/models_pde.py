import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 4):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Tanh())
            return layers
        
        self.layers = block(in_dim, hid_dim)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 2):
        super(Discriminator, self).__init__()
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Tanh())
            return layers

        self.layers = block(in_dim, hid_dim)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
        
class Q_Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 4):
        super(Q_Net, self).__init__()
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Tanh())
            return layers
        
        self.layers = block(in_dim, hid_dim)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
    
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 4, rate = 0.2):
        super(Net, self).__init__()
        
        def block(in_feat, out_feat, dropout = False):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.Tanh())
            if dropout:
                layers.append(nn.Dropout(rate))
            return layers
        
        self.layers = block(in_dim, hid_dim, dropout=False)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim, dropout = True)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
        