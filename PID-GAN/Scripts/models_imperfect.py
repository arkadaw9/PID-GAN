import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 4):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, isBatch=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if isBatch:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.layers = block(in_dim, hid_dim, False)
            
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
        
        def block(in_feat, out_feat, isBatch=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if isBatch:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.layers = block(in_dim, hid_dim, False)
            
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
        
        def block(in_feat, out_feat, isBatch=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if isBatch:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.layers = block(in_dim, hid_dim, False)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out
    

class PGNN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim = 50, num_layers = 4):
        super(PGNN, self).__init__()
        
        def block(in_feat, out_feat, isBatch=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if isBatch:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.2))
            return layers
        
        self.layers = block(in_dim, hid_dim, False)
            
        for layer_i in range(num_layers - 1):
            self.layers += block(hid_dim, hid_dim)
                
        self.layers.append(nn.Linear(hid_dim, out_dim))
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        out = self.model(x)
        return out