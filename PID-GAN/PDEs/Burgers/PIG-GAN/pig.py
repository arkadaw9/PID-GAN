import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Burgers_PIG():
    def __init__(self, x_u, y_u, x_f, X_star, u_star, G, D, Q, device, nepochs, lambda_phy, noise = 0.0):
        super(Burgers_PIG, self).__init__()
        
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
        
        self.nepochs = nepochs
        self.lambda_phy = lambda_phy
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
    
    def sample_noise(self, number, size=1):
        noises = torch.randn((number, size)).float().to(self.device)
        return noises
    
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
    
    def n_phy_prob(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=X.shape[0])
        u = self.G(torch.cat([x, t, noise], dim=1))
        residual = self.phy_residual(x, t, u)
        return residual, u, noise
    
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=x.shape[0])
        u = self.G(torch.cat([x, t, noise], dim=1))
        f = self.phy_residual(x, t, u)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f
    
    def average_prediction(self, nsamples = 500):
        u_pred_list = []
        f_pred_list = []
        for run in range(nsamples):
            u_pred, f_pred = self.predict(self.X_star_norm)
            u_pred_list.append(u_pred)
            f_pred_list.append(f_pred)

        u_pred_arr = np.array(u_pred_list)
        f_pred_arr = np.array(f_pred_list)

        u_pred = u_pred_arr.mean(axis=0)
        f_pred = f_pred_arr.mean(axis=0)

        u_dev = u_pred_arr.var(axis=0)
        f_dev = f_pred_arr.var(axis=0)

        error_u = np.linalg.norm(self.u_star-u_pred,2)/np.linalg.norm(self.u_star,2)
        residual = (f_pred**2).mean()

#         U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

        return error_u, residual, u_pred
    
    def train_disriminator(self, x, y):
        # training discriminator...
        self.D_optimizer.zero_grad()
        
        real_logits = self.D.forward(torch.cat([x, y], dim=1))

        # physics loss for boundary points
        _, u, _ = self.n_phy_prob(x)
        fake_logits_u = self.D.forward(torch.cat([x, u], dim=1))

        # discriminator loss
        d_loss = self.discriminator_loss(real_logits, fake_logits_u)

        d_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        
        return d_loss
    
    def train_generator(self, x, y):
        # training generator...
        for gen_epoch in range(5):
            self.G_optimizer.zero_grad()
            # physics loss for collocation points
            phyloss_1, _, _ = self.n_phy_prob(self.train_x_f)
            
            # physics loss for boundary points
            _, y_pred, G_noise = self.n_phy_prob(x)
            fake_logits_u = self.D.forward(torch.cat([x, y_pred], dim=1))

            z_pred = self.Q.forward(torch.cat([x, y_pred], dim=1))
            mse_loss_Z = torch.nn.functional.mse_loss(z_pred, G_noise)

            mse_loss = torch.nn.functional.mse_loss(y_pred, y)
            adv_loss = self.generator_loss(fake_logits_u)
            
            phy_loss = (phyloss_1**2).mean()

            g_loss = adv_loss + self.lambda_phy * phy_loss + self.lambda_q * mse_loss_Z


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
                error_u, residual, _ = self.average_prediction()
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [G loss: %f] [D loss: %f] [Q loss: %f] [Adv G loss: %f] [Err u: %e] [Residual_f: %e]"
                    % (epoch, self.nepochs, MSE_loss[epoch], G_loss[epoch], D_loss[epoch], Q_loss[epoch], Adv_loss[epoch], error_u, residual )
                )