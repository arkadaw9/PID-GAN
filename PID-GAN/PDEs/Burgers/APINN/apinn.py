import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Burgers_APINN():
    def __init__(self, x_u, y_u, x_f, X_star, u_star, net, device, nepochs, lambda_mse, noise = 0.0):
        super(Burgers_APINN, self).__init__()
        
        # Normalize data
        self.Xmean, self.Xstd = x_f.mean(0), x_f.std(0)
        self.x_f = (x_f - self.Xmean) / self.Xstd
        self.x_u = (x_u - self.Xmean) / self.Xstd
        self.X_star_norm = (X_star - self.Xmean) / self.Xstd
        self.u_star = u_star
        
        #Jacobian of the PDE because of normalization
        self.Jacobian_X = 1 / self.Xstd[0]
        self.Jacobian_T = 1 / self.Xstd[1]
        
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.net = net
        
        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=1e-4, betas = (0.9, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        self.nepochs = nepochs
        self.lambda_mse = lambda_mse
        
        self.batch_size = 150
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )
        
    def get_residual(self, X):
        # physics loss for collocation/boundary points
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        u = self.net.forward(torch.cat([x, t], dim=1))
        f = self.phy_residual(x, t, u)
        return u, f
    
    def uncertainity_estimate(self, x, num_samples=500):
        outputs = np.hstack([self.net(x).cpu().detach().numpy() for i in range(num_samples)]) 
        y_variance = outputs.var(axis=1)
        y_std = np.sqrt(y_variance)
        return y_mean, y_std
    
    def phy_residual(self, x, t, u, nu = (0.01/np.pi)):
        """ The pytorch autograd version of calculating residual """

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = (self.Jacobian_T) * u_t + (self.Jacobian_X) * u * u_x - nu * (self.Jacobian_X ** 2) * u_xx 
        return f
    
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
        TOT_loss = np.zeros(self.nepochs)
        MSE_loss = np.zeros(self.nepochs)
        PHY_loss = np.zeros(self.nepochs)
        

        for epoch in range(self.nepochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(self.train_loader):


                self.net_optim.zero_grad()

                y_pred, _ = self.get_residual(x)
                _, residual = self.get_residual(self.train_x_f)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(residual**2)
                
                self.lambda_mse = self.find_lambda(self.lambda_mse, physics_loss, self.lambda_mse * mse_loss)

                loss = self.lambda_mse * mse_loss + physics_loss

                loss.backward(retain_graph=True)
                self.net_optim.step()


                TOT_loss[epoch] += loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                PHY_loss[epoch] += physics_loss.detach().cpu().numpy()


            TOT_loss[epoch] = TOT_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            PHY_loss[epoch] = PHY_loss[epoch] / len(self.train_loader)


            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f] [Lambda_mse: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch], self.lambda_mse)
                )