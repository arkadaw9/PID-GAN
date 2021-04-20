import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
from scipy.interpolate import griddata
from torch.utils.data import TensorDataset, DataLoader



class Darcy_PIG:
    def __init__(self, x_u, y_u, x_b, x_f, 
                 N_u, G, kG, D, Q, device, 
                 num_epochs, lambdas, noise=0.0):
        
        super(Darcy_PIG, self).__init__()
        
        # adding noise to the labeled point
        self.y_u = y_u + noise * np.std(y_u)*np.random.randn(y_u.shape[0], y_u.shape[1])
        
        # labeled point
        self.train_x_u = torch.tensor(x_u, requires_grad=True).float().to(device)
        self.train_y_u = torch.tensor(y_u, requires_grad=True).float().to(device)
        # boundary point
        self.train_x_b1 = torch.tensor(x_b[:, 0:2], requires_grad=True).float().to(device)
        self.train_x_b2 = torch.tensor(x_b[:, 2:4], requires_grad=True).float().to(device)
        self.train_x_b3 = torch.tensor(x_b[:, 4:6], requires_grad=True).float().to(device)
        self.train_x_b4 = torch.tensor(x_b[:, 6:8], requires_grad=True).float().to(device)
        # collocation point
        self.train_x_f = torch.tensor(x_f, requires_grad=True).float().to(device)
        
        # train loader
        batch_size = N_u
        shuffle = True
        self.train_loader = DataLoader(
            list(zip(self.train_x_u, self.train_y_u)), 
            batch_size=batch_size, shuffle=shuffle
        )
        
        # models
        self.G = G
        self.kG = kG
        self.D = D
        self.Q = Q
        
        self.device = device
        self.num_epochs = num_epochs
        self.lambda_phy = lambdas[0]
        self.lambda_q = lambdas[1]
        
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-4, betas = (0.9, 0.999))
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-4, betas = (0.9, 0.999))
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-4, betas = (0.9, 0.999))
        self.kG_optimizer = torch.optim.Adam(self.kG.parameters(), lr=1e-4, betas = (0.9, 0.999))
        
    def sample_noise(self, number, size=2):
        noises = torch.randn((number, size)).float().to(self.device)
        return noises

    def discriminator_loss(self, logits_real_u, logits_fake_u):

        loss = -torch.mean(torch.log(1.0 - torch.sigmoid(logits_real_u) + 1e-8) + \
                           torch.log(torch.sigmoid(logits_fake_u) + 1e-8)) 
        return loss

    def generator_loss(self, logits_fake_u):

        gen_loss = torch.mean(logits_fake_u)

        return gen_loss

    def boundary_1_loss(self, train_x_b1, generator, k_generator, q = 1.):
        z_prior = self.sample_noise(number = train_x_b1.shape[0]) 
        x1 = torch.tensor(train_x_b1[:,0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(train_x_b1[:,1:2], requires_grad=True).float().to(self.device)
        u = generator(torch.cat([x1, x2, z_prior], dim=1))
        u_x1 = torch.autograd.grad(
                u, x1, 
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
        )[0]
        k = k_generator(u)
        temp = q + k * u_x1
        return (temp**2).mean()

    def boundary_2_loss(self, train_x_b2, generator):
        z_prior = self.sample_noise(number = train_x_b2.shape[0]) 
        x1 = torch.tensor(train_x_b2[:,0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(train_x_b2[:,1:2], requires_grad=True).float().to(self.device)
        u = generator(torch.cat([x1, x2, z_prior], dim=1))
        u_x2 = torch.autograd.grad(
                u, x2, 
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
        )[0]
        return (u_x2**2).mean()

    def boundary_3_loss(self, train_x_b3, generator, u_0= -10):
        z_prior = self.sample_noise(number = train_x_b3.shape[0]) 
        x1 = torch.tensor(train_x_b3[:,0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(train_x_b3[:,1:2], requires_grad=True).float().to(self.device)
        u = generator(torch.cat([x1, x2, z_prior], dim=1))
        temp = u - u_0
        return (temp**2).mean()

    def boundary_4_loss(self, train_x_b4, generator):
        z_prior = self.sample_noise(number = train_x_b4.shape[0]) 
        x1 = torch.tensor(train_x_b4[:,0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(train_x_b4[:,1:2], requires_grad=True).float().to(self.device)
        u = generator(torch.cat([x1, x2, z_prior], dim=1))
        u_x2 = torch.autograd.grad(
                u, x2, 
                grad_outputs=torch.ones_like(u),
                retain_graph=True,
                create_graph=True
        )[0]
        return (u_x2**2).mean()

    def phy_residual(self, x1, x2, u, k):
        """ The pytorch autograd version of calculating residual """
        u_x1 = torch.autograd.grad(
            u, x1, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x2 = torch.autograd.grad(
            u, x2, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        f_1 = torch.autograd.grad(
            k*u_x1, x1, 
            grad_outputs=torch.ones_like(u_x1),
            retain_graph=True,
            create_graph=True
        )[0]
        f_2 = torch.autograd.grad(
            k*u_x2, x2, 
            grad_outputs=torch.ones_like(u_x2),
            retain_graph=True,
            create_graph=True
        )[0]
        f = f_1 + f_2
        return f

    def n_phy_prob(self, train_x, generator, k_generator, lambda_prob=0.2):
        # physics loss for collocation/boundary points
        x1 = torch.tensor(train_x[:, 0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(train_x[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=train_x.shape[0])

        u = generator(torch.cat([x1, x2, noise], dim=1))
        k = k_generator(u)
        phyloss = self.phy_residual(x1, x2, u, k)
        n_phy = torch.exp(-lambda_prob * (phyloss**2))
        return n_phy, phyloss, u, noise

    def predict(self, X):
        x1 = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        x2 = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        noise = self.sample_noise(number=x1.shape[0])

        self.G.eval()
        self.kG.eval()
        u = self.G(torch.cat([x1, x2, noise], dim=1))
        k = self.kG(u)
        f = self.phy_residual(x1, x2, u, k)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        k = k.detach().cpu().numpy()
        return u, f, k
        
    def train_disriminator(self, x, y):
        
        # training discriminator...
        self.D_optimizer.zero_grad()
        
        # computing real logits for discriminator loss
        real_logits = self.D.forward(torch.cat([x, y], dim=1))
        
        # physics loss for boundary points
        _, _, u, _ = self.n_phy_prob(x, self.G, self.kG)
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
            self.kG_optimizer.zero_grad()
            
            # physics loss for collocation points
            _, phyloss, _, _ = self.n_phy_prob(self.train_x_f, self.G, self.kG)

            # physics loss for labeled points
            _, _, y_pred, G_noise = self.n_phy_prob(x, self.G, self.kG)
            fake_logits_u = self.D.forward(torch.cat([x, y_pred], dim=1))


            z_pred = self.Q.forward(torch.cat([x, y_pred], dim=1))
            mse_loss_Z = torch.nn.functional.mse_loss(z_pred, G_noise)

            mse_loss = torch.nn.functional.mse_loss(y_pred, y)
            adv_loss = self.generator_loss(fake_logits_u)
            
            boundary_loss = self.boundary_1_loss(self.train_x_b1, self.G, self.kG, q = 1.) +\
                            self.boundary_2_loss(self.train_x_b2, self.G) +\
                            self.boundary_3_loss(self.train_x_b3, self.G, u_0= -10) +\
                            self.boundary_4_loss(self.train_x_b4, self.G)

            phy_loss = (phyloss**2).mean()

            g_loss = adv_loss + self.lambda_phy * phy_loss + self.lambda_q * mse_loss_Z +\
                     boundary_loss

            g_loss.backward(retain_graph=True)
            
            self.G_optimizer.step()
            self.kG_optimizer.step()
            
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
        Adv_loss = np.zeros(self.num_epochs)
        G_loss = np.zeros(self.num_epochs)
        D_loss = np.zeros(self.num_epochs)
        Q_loss = np.zeros(self.num_epochs)

        MSE_loss = np.zeros(self.num_epochs)

        G_loss_batch = []
        D_loss_batch = []
        

        for epoch in range(self.num_epochs):
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
                print(
                    "[Epoch %d/%d] [MSE loss: %f] [G loss: %f] [D loss: %f] [Q loss: %f] [Adv G loss: %f]"
                    % (epoch, self.num_epochs, MSE_loss[epoch], G_loss[epoch], D_loss[epoch], Q_loss[epoch], Adv_loss[epoch])
                )
        
        
        

