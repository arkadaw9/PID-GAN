import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Schrodinger_PIG():
    def __init__(self, x_u, y_u, x_f, x_lb, x_ub, X_star, h_star, G, D, Q, device, nepochs, lambda_val, noise = 0.0):
        super(Schrodinger_PIG, self).__init__()
        
        self.x_f = x_f
        self.x_u = x_u 
        self.X_star = X_star
        self.h_star = h_star
        
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        self.G = G
        self.D = D
        self.Q = Q
        
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas = (0.9, 0.999))
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas = (0.9, 0.999))
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-4, betas = (0.9, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x_u = torch.tensor(self.x_u, requires_grad=True).float().to(self.device)
        self.train_y_u = torch.tensor(self.y_u, requires_grad=True).float().to(self.device)
        self.train_x_f = torch.tensor(self.x_f, requires_grad=True).float().to(self.device)
        
        #Boundary conditions
        self.train_x_lb = torch.tensor(x_lb, requires_grad=True).float().to(device)
        self.train_x_ub = torch.tensor(x_ub, requires_grad=True).float().to(device)

        self.nepochs = nepochs
        self.lambda_val = lambda_val
        self.lambda_q = 0.5
        
        self.batch_size = 150
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u,self.train_y_u)), batch_size=self.batch_size, shuffle=shuffle
        )
        
    def discriminator_loss(self, logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8))
        return loss

    def generator_loss(self, logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss
    
    def boundary_loss(self):
        x_lb = torch.tensor(self.train_x_lb[:, 0:1], requires_grad=True).float().to(self.device)
        t_lb = torch.tensor(self.train_x_lb[:, 1:2], requires_grad=True).float().to(self.device)

        x_ub = torch.tensor(self.train_x_ub[:, 0:1], requires_grad=True).float().to(self.device)
        t_ub = torch.tensor(self.train_x_ub[:, 1:2], requires_grad=True).float().to(self.device)

        noise_lb = self.sample_noise(number = self.train_x_lb.shape[0])
        noise_ub = self.sample_noise(number = self.train_x_ub.shape[0])
        y_pred_lb = self.G.forward(torch.cat([x_lb, t_lb, noise_lb], dim=1))
        y_pred_ub = self.G.forward(torch.cat([x_ub, t_ub, noise_ub], dim=1))
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
    
    def sample_noise(self, number, size=1):
        noises = torch.randn((number, size)).float().to(self.device)
        return noises
    
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
    
    def n_phy_prob(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number = X.shape[0])
        
        y_pred = self.G.forward(torch.cat([x, t, noise], dim=1))
        u = y_pred[:,0:1]
        v = y_pred[:,1:2]
        f_u, f_v = self.phy_residual(x, t, u, v)
        return f_u, f_v, y_pred, noise
    
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=x.shape[0])
        y_pred = self.G(torch.cat([x, t, noise], dim=1))
        u = y_pred[:,0:1]
        v = y_pred[:,1:2]
        
        f_u, f_v = self.phy_residual(x, t, u, v)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        f_u = f_u.detach().cpu().numpy()
        f_v = f_v.detach().cpu().numpy()
        return u, v, f_u, f_v
    
    def average_prediction(self, nsamples = 500):
        u_pred_list = []
        v_pred_list = []
        h_pred_list = []
        f_u_pred_list = []
        f_v_pred_list = []
        for run in range(nsamples):
            u_pred, v_pred, f_u_pred, f_v_pred = self.predict(self.X_star)
            h_pred = np.sqrt(u_pred**2 + v_pred**2)
            u_pred_list.append(u_pred)
            v_pred_list.append(v_pred)
            f_u_pred_list.append(f_u_pred)
            f_v_pred_list.append(f_v_pred)
            h_pred_list.append(h_pred)


        u_pred_arr = np.array(u_pred_list)
        v_pred_arr = np.array(v_pred_list)
        f_u_pred_arr = np.array(f_u_pred_list)
        f_v_pred_arr = np.array(f_v_pred_list)
        h_pred_arr = np.array(h_pred_list)

        u_pred = u_pred_arr.mean(axis=0)
        v_pred = v_pred_arr.mean(axis=0)
        f_u_pred = f_u_pred_arr.mean(axis=0)
        f_v_pred = f_v_pred_arr.mean(axis=0)
        h_pred = h_pred_arr.mean(axis=0)

        h_pred_var = h_pred_arr.var(axis=0)
        
        error_h = np.linalg.norm(self.h_star-h_pred,2)/np.linalg.norm(self.h_star,2)
        
        residual = (f_u_pred**2).mean() + (f_v_pred**2).mean()
        
        return error_h, residual
    
    def train_disriminator(self, x, y):
        # training discriminator...
        self.D_optimizer.zero_grad()
        
        # computing real logits for discriminator loss
        real_logits_x0 = self.D.forward(torch.cat([x, y], dim=1))
        
        # physics loss for boundary points
        _, _, y_pred_x0, _ = self.n_phy_prob(x)
        fake_logits_x0 = self.D.forward(torch.cat([x, y_pred_x0], dim=1))
        
        # discriminator loss
        d_loss = self.discriminator_loss(real_logits_x0, fake_logits_x0)

        d_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        return d_loss
    
    def train_generator(self, x, y):
        # training generator...
        
        for gen_epoch in range(5):
        
            self.G_optimizer.zero_grad()
            
            # physics loss for collocation points
            phyloss_u_xf, phyloss_v_xf, y_pred_xf, _ = self.n_phy_prob(self.train_x_f)

            # physics loss for boundary points
            _, _, y_pred_x0, G_noise = self.n_phy_prob(x)
            fake_logits_x0 = self.D.forward(torch.cat([x, y_pred_x0], dim=1))


            z_pred = self.Q.forward(torch.cat([x, y_pred_x0], dim=1))
            mse_loss_Z = torch.nn.functional.mse_loss(z_pred, G_noise)

            mse_loss = torch.nn.functional.mse_loss(y_pred_x0, y)
            adv_loss = self.generator_loss(fake_logits_x0)
      
            b_loss = self.boundary_loss()
            
            phy_loss = (phyloss_u_xf**2).mean() + (phyloss_v_xf**2).mean()

            g_loss = adv_loss + self.lambda_val * phy_loss + self.lambda_q * mse_loss_Z + b_loss

            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()
            
            return g_loss, adv_loss, mse_loss
    
    def train_qnet(self, x, y):
        self.Q_optimizer.zero_grad()
        Q_noise = self.sample_noise(number=x.shape[0])
        y_pred = self.G.forward(torch.cat([x, Q_noise], dim=1))
        z_pred = self.Q.forward(torch.cat([x, y_pred], dim=1))
        q_loss = torch.nn.functional.mse_loss(z_pred, Q_noise)
        q_loss.backward()
        self.Q_optimizer.step()
        return q_loss
    
    def train(self):
        Adv_loss = np.zeros(self.nepochs)
        G_loss = np.zeros(self.nepochs)
        D_loss = np.zeros(self.nepochs)
        Q_loss = np.zeros(self.nepochs)

        MSE_loss = np.zeros(self.nepochs)

        G_loss_batch = []
        D_loss_batch = []
        

        for epoch in range(self.nepochs):
            epoch_loss = 0        
            for i, (x, y) in enumerate(self.train_loader):

                d_loss = self.train_disriminator(x,y)
                
                g_loss, adv_loss, mse_loss = self.train_generator(x,y)
                
                q_loss = self.train_qnet(x,y)
                

                G_loss_batch.append(g_loss.detach().cpu().numpy())
                D_loss_batch.append(d_loss.detach().cpu().numpy())

                Adv_loss[epoch] += adv_loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                G_loss[epoch] += g_loss.detach().cpu().numpy()
                D_loss[epoch] += d_loss.detach().cpu().numpy()
                Q_loss[epoch] += q_loss.detach().cpu().numpy()


            Adv_loss[epoch] = Adv_loss[epoch] / len(self.train_loader)
            MSE_loss[epoch] = MSE_loss[epoch] / len(self.train_loader)
            G_loss[epoch] = G_loss[epoch] / len(self.train_loader)
            D_loss[epoch] = D_loss[epoch] / len(self.train_loader)
            Q_loss[epoch] = Q_loss[epoch] / len(self.train_loader)


            if (epoch % 100 == 0):
                error_h, residual = self.average_prediction()
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [G loss: %f] [D loss: %f] [Q loss: %f] [Adv G loss: %f] [Err h: %e] [Residual_f: %e]"
                    % (epoch, self.nepochs, MSE_loss[epoch], G_loss[epoch], D_loss[epoch], Q_loss[epoch], Adv_loss[epoch], error_h, residual )
                )