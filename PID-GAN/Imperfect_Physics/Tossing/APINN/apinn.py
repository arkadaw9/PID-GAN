import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Tossing_APINN():
    def __init__(self, train_x, train_y, test_x, test_y, net, device, nepochs, lambda_val):
        super(Tossing_APINN, self).__init__()
        
        # Normalize data
        self.Xmean, self.Xstd = train_x.mean(0), train_x.std(0)
        self.Ymean, self.Ystd = train_y.mean(0), train_y.std(0)
        
        self.train_x = (train_x - self.Xmean) / self.Xstd
        self.test_x = (test_x - self.Xmean) / self.Xstd
        self.train_y = (train_y- self.Ymean) / self.Ystd
        self.test_y = (test_y - self.Ymean) / self.Ystd
        
        self.net = net
        
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3, betas = (0.5, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x = torch.tensor(self.train_x, requires_grad=True).float().to(self.device)
        self.train_y = torch.tensor(self.train_y, requires_grad=True).float().to(self.device)
        self.test_x = torch.tensor(self.test_x, requires_grad=True).float().to(self.device)
        self.test_y = torch.tensor(self.test_y, requires_grad=True).float().to(self.device)
        self.x_f = torch.cat([self.train_x, self.test_x], dim = 0)
        
        self.nepochs = nepochs
        self.lambda_mse = lambda_val
        
        self.batch_size = 64
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x,self.train_y)), batch_size=self.batch_size, shuffle=shuffle
        )
    
    def uncertainity_estimate(self, x, num_samples, stat = [0,1]):
        outputs = np.stack([self.net.forward(x).cpu().detach().numpy()*stat[1]+stat[0] for i in range(num_samples)], axis = 0)
        y_mean = outputs.mean(axis=0)
        y_variance = outputs.var(axis=0)
        y_std = np.sqrt(y_variance)
        return y_mean, y_std
    
    def physics_loss(self, inp, out, stat_inp = [0,1], stat_out = [0,1], t = 0.0333333):  #stat [0] = mean, stat [1] = std
    
        fps = 30
        stat_inp = torch.Tensor(stat_inp).to(self.device)
        stat_out = torch.Tensor(stat_out).to(self.device)
        inp = inp * stat_inp[1] + stat_inp[0]
        out = out * stat_out[1] + stat_out[0]


        v_x = (inp[:,1] - inp[:, 0])/t
        v_y = (inp[:,4] - inp[:, 3] + 0.5 * 9.8 * t*t)/t

        pred_x = out[:, 0:15]
        pred_y = out[:, 15:]

        g = 9.8

        t = np.arange(0, 1/fps * (inp.shape[1] + out.shape[1]) * 0.5, 1/fps)

        t = np.repeat([t[3:]], inp.shape[0], axis=0)
        t_s = torch.tensor(t).float().to(self.device)

        x_0 = inp[:, 0].repeat(15, 1).T
        y_0 = inp[:, 3].repeat(15, 1).T
        v_x_t = v_x.repeat(15, 1).T
        v_y_t = v_y.repeat(15, 1).T

        x_loc_pred = x_0 + v_x_t * t_s
        y_loc_pred = y_0 + v_y_t * t_s - 0.5 * g * t_s**2


        loss = 0.5*(((pred_x  - x_loc_pred)**2).mean(dim=1) + ((pred_y  - y_loc_pred)**2).mean(dim=1))
        return loss
    
    def find_lambda(self, adaptive_lambda, phy_loss, loss, beta = 0.9):
        phyloss_layer = []
        loss_layer = []
        with torch.no_grad():
            for layer in self.net.model.children():
                if isinstance(layer, nn.Linear):
                    phyloss_layer.append(torch.abs(torch.autograd.grad(phy_loss, layer.weight, retain_graph = True)[0]).max())
                    loss_layer.append(torch.abs(torch.autograd.grad(loss, layer.weight, retain_graph = True)[0]).mean())
        max_grad_res = torch.stack(phyloss_layer).max()     
        mean_grad_loss = torch.stack(loss_layer).mean()
        lambda_new = max_grad_res / mean_grad_loss
        adaptive_lambda = (1 - beta) * adaptive_lambda + beta * lambda_new
        return adaptive_lambda
    
    def train(self):
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)
        TOT_loss = np.zeros(self.nepochs)

        for epoch in range(self.nepochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(self.train_loader):

                self.net_optimizer.zero_grad()
                y_pred = self.net.forward(x)

                y_f = self.net.forward(self.x_f)

                phy_loss = torch.mean(torch.abs(self.physics_loss(self.x_f, y_f,  [self.Xmean, self.Xstd], [self.Ymean, self.Ystd])))
                mse_loss = torch.nn.functional.mse_loss(y_pred, y)
                
                self.lambda_mse = self.find_lambda(self.lambda_mse, phy_loss, self.lambda_mse * mse_loss)
                
                loss = self.lambda_mse*mse_loss + phy_loss
                loss.backward()
                self.net_optimizer.step()

                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += phy_loss.detach().cpu().numpy()
                TOT_loss[epoch] += loss.detach().cpu().numpy()

            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)
            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)

            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f] [Lambda mse : %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch], self.lambda_mse )
            )