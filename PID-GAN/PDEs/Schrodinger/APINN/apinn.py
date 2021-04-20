import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Schrodinger_APINN():
    def __init__(self, x_u, y_u, x_f, x_lb, x_ub, X_star, h_star, net, device, nepochs, noise = 0.0):
        super(Schrodinger_APINN, self).__init__()
        
        self.x_f = x_f
        self.x_u = x_u 
        self.X_star = X_star
        self.h_star = h_star
        
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.net = net
        
        self.net_optim = torch.optim.Adam(self.net.parameters(), lr=1e-4, betas = (0.9, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        #Boundary conditions
        self.train_x_lb = torch.tensor(x_lb, requires_grad=True).float().to(device)
        self.train_x_ub = torch.tensor(x_ub, requires_grad=True).float().to(device)

        self.nepochs = nepochs
        self.lambda_mse = 1
        self.lambda_boundary = 1
        self.lambda_q = 0.5
        
        self.batch_size = 150
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )
    
    def boundary_loss(self):
        x_lb = torch.tensor(self.train_x_lb[:, 0:1], requires_grad=True).float().to(self.device)
        t_lb = torch.tensor(self.train_x_lb[:, 1:2], requires_grad=True).float().to(self.device)

        x_ub = torch.tensor(self.train_x_ub[:, 0:1], requires_grad=True).float().to(self.device)
        t_ub = torch.tensor(self.train_x_ub[:, 1:2], requires_grad=True).float().to(self.device)
        
        y_pred_lb = self.net.forward(torch.cat([x_lb, t_lb], dim=1))
        y_pred_ub = self.net.forward(torch.cat([x_ub, t_ub], dim=1))
        
        u_lb = y_pred_lb[:,0:1]
        v_lb = y_pred_lb[:,1:2]
        u_ub = y_pred_ub[:,0:1]
        v_ub = y_pred_ub[:,1:2]

        u_lb_x = torch.autograd.grad(
                u_lb, x_lb, 
                grad_outputs=torch.ones_like(u_lb),
                retain_graph=True,
                create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
                u_ub, x_ub, 
                grad_outputs=torch.ones_like(u_ub),
                retain_graph=True,
                create_graph=True
            )[0]

        v_lb_x = torch.autograd.grad(
                v_lb, x_lb, 
                grad_outputs=torch.ones_like(v_lb),
                retain_graph=True,
                create_graph=True
            )[0]

        v_ub_x = torch.autograd.grad(
                v_ub, x_ub, 
                grad_outputs=torch.ones_like(v_ub),
                retain_graph=True,
                create_graph=True
            )[0]

        loss = torch.mean((y_pred_lb - y_pred_ub)**2) + \
               torch.mean((u_lb_x - u_ub_x)**2) + \
               torch.mean((v_lb_x - v_ub_x)**2)
        return loss
    
    def phy_residual(self, x, t, u, v):
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
        
        v_t = torch.autograd.grad(
            v, t, 
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        
        v_x = torch.autograd.grad(
            v, x, 
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        
        v_xx = torch.autograd.grad(
            v_x, x, 
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
        return f_u, f_v
    
    def get_residual(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        y_pred = self.net.forward(torch.cat([x, t], dim=1))
        u = y_pred[:,0:1]
        v = y_pred[:,1:2]
        f_u, f_v = self.phy_residual(x, t, u, v)
        return y_pred, f_u, f_v
    
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

                y_pred, _, _ = self.get_residual(x)
                _, f_u, f_v  = self.get_residual(self.train_x_f)

                mse_loss = torch.nn.functional.mse_loss(y, y_pred)
                physics_loss = torch.mean(f_u**2 + f_v**2)
                
                b_loss = self.boundary_loss()
                
                self.lambda_mse = self.find_lambda(self.lambda_mse, physics_loss, self.lambda_mse * mse_loss)
                self.lambda_boundary = self.find_lambda(self.lambda_boundary, physics_loss, self.lambda_boundary * b_loss)

                loss = self.lambda_mse * mse_loss + physics_loss + self.lambda_boundary * b_loss

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
                    "[Epoch %d/%d] [MSE loss: %f] [Phy loss: %f] [Total loss: %f] [Lambda mse : %f] [Lambda boundary : %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], PHY_loss[epoch], TOT_loss[epoch], self.lambda_mse, self.lambda_boundary)
                )