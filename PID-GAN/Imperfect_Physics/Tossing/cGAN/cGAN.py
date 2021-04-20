import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader

class Tossing_cGAN():
    def __init__(self, train_x, train_y, test_x, test_y, G, D, Q, device, nepochs, lambda_val, noise_dim):
        super(Tossing_cGAN, self).__init__()
        
        # Normalize data
        self.Xmean, self.Xstd = train_x.mean(0), train_x.std(0)
        self.Ymean, self.Ystd = train_y.mean(0), train_y.std(0)
        
        self.train_x = (train_x - self.Xmean) / self.Xstd
        self.test_x = (test_x - self.Xmean) / self.Xstd
        self.train_y = (train_y- self.Ymean) / self.Ystd
        self.test_y = (test_y - self.Ymean) / self.Ystd
        
        self.G = G
        self.D = D
        self.Q = Q
        
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3, betas = (0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas = (0.5, 0.999))
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3, betas = (0.5, 0.999))
        
        self.device = device
        
        # numpy to tensor
        self.train_x = torch.tensor(self.train_x, requires_grad=True).float().to(self.device)
        self.train_y = torch.tensor(self.train_y, requires_grad=True).float().to(self.device)
        self.test_x = torch.tensor(self.test_x, requires_grad=True).float().to(self.device)
        self.test_y = torch.tensor(self.test_y, requires_grad=True).float().to(self.device)
        self.x_f = torch.cat([self.train_x, self.test_x], dim = 0)
        
        self.nepochs = nepochs
        self.lambda_val = 0 #for cGAN we put lambda = 0
        self.lambda_q = 0.5
        self.noise_dim = noise_dim
        
        self.batch_size = 64
        num_workers = 4
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x,self.train_y)), batch_size=self.batch_size, shuffle=shuffle
        )
        
    def discriminator_loss(self, logits_real_u, logits_fake_u):
        loss =  - torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

    def generator_loss(self, logits_fake_u):
        gen_loss = torch.mean(logits_fake_u)
        return gen_loss
    
    def sample_noise(self, batch_size, dim, mean=0, std=1):
        to_return = mean + std * torch.randn((batch_size, dim))
        return to_return
    
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
    
    def train_discriminator(self, x, y):
        # DISCRIMINATOR UPDATE
        self.D_optimizer.zero_grad()

        ## REAL DATA         
        real_logits = self.D.forward(torch.cat([x, y], dim=1))

        ## PREDICTED SAMPLES FROM BATCH
        D_noise = self.sample_noise(x.shape[0], self.noise_dim).to(self.device)
        y_pred = self.G.forward(torch.cat([x, D_noise], dim=1))
        fake_logits = self.D.forward(torch.cat([x, y_pred], dim=1)) 

        d_loss = self.discriminator_loss(real_logits, fake_logits)

        d_loss.backward(retain_graph = True)
        self.D_optimizer.step()
        
        return d_loss
    
    def train_generator(self, x, y):
        # GENERATOR UPDATE
        self.G_optimizer.zero_grad()

        G_noise = self.sample_noise(x.shape[0], self.noise_dim).to(self.device)
        y_pred = self.G.forward(torch.cat([x, G_noise], dim=1))
        fake_logits = self.D.forward(torch.cat([x, y_pred], dim=1))

        ## UNLABELLED DATA SAMPLES
        G_noise_f = self.sample_noise(self.x_f.shape[0], self.noise_dim).to(self.device)
        y_pred_f = self.G.forward(torch.cat([self.x_f, G_noise_f], dim=1))
        phy_loss = torch.abs(self.physics_loss(self.x_f, y_pred_f, [self.Xmean, self.Xstd], [self.Ymean, self.Ystd])).mean()
            
        mse_loss = torch.nn.functional.mse_loss(y_pred, y)
            
        z_pred = self.Q.forward(torch.cat([x, y_pred], dim=1))
        mse_loss_Z = torch.nn.functional.mse_loss(z_pred, G_noise)

        adv_loss = self.generator_loss(fake_logits) 
        g_loss = adv_loss + self.lambda_val * phy_loss + self.lambda_q * mse_loss_Z

        g_loss.backward(retain_graph = True)
        self.G_optimizer.step()
            
        return g_loss, adv_loss, mse_loss, phy_loss
    
    def train_qnet(self, x, y):
        self.Q_optimizer.zero_grad()
        Q_noise = self.sample_noise(x.shape[0], self.noise_dim).to(self.device)
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
        PHY_loss = np.zeros(self.nepochs)

        G_loss_batch = []
        D_loss_batch = []

        for epoch in range(self.nepochs):
            epoch_loss = 0
            for i, (x, y) in enumerate(self.train_loader):

                d_loss = self.train_discriminator(x, y)
                
                g_loss, adv_loss, mse_loss, phy_loss = self.train_generator(x, y)
                
                q_loss = self.train_qnet(x , y)

                G_loss_batch.append(g_loss.detach().cpu().numpy())
                D_loss_batch.append(d_loss.detach().cpu().numpy())

                Adv_loss[epoch] += adv_loss.detach().cpu().numpy()
                MSE_loss[epoch] += mse_loss.detach().cpu().numpy()
                G_loss[epoch] += g_loss.detach().cpu().numpy()
                D_loss[epoch] += d_loss.detach().cpu().numpy()
                Q_loss[epoch] += q_loss.detach().cpu().numpy()
                PHY_loss[epoch] += phy_loss.detach().cpu().numpy()

            if (epoch % 100 == 0):
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [G loss: %f] [D loss: %f] [Q loss: %f] [Phy loss: %f] [Adv G loss: %f]"
                    % (epoch, self.nepochs, MSE_loss[epoch], G_loss[epoch], D_loss[epoch], Q_loss[epoch], PHY_loss[epoch], Adv_loss[epoch] )
                )