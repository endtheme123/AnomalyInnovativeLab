import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18
from my_vit import ViT

class ViT_VAE(nn.Module):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
        '''
        '''
        super(ViT_VAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta
        self.rec_loss = rec_loss
        self.delta = delta
        self.dropout = nn.Dropout(p=0.5)

        # [b, 784] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.vit = ViT(
                        image_size = 224,
                        patch_size = 16,
                        num_classes = 32,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 2048,
                        dropout = 0.1,
                        emb_dropout = 0.1
        )


        
        self.linear_up = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 3*224*224),

        )

    def encoder(self, x):
        x = self.vit(x)
        
        return x[:, :16], x[:, 16:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        x = self.linear_up(z)
        
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        print(mu.shape)
        self.logvar = logvar
        x_rec = self.decoder(z)
        x_rec = x_rec.reshape(8,3,224,224)
        return x_rec, (mu, logvar)

    def xent_continuous_ber(self, recon_x, x, pixelwise=False):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return (x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x))
        else:
            return torch.sum(x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x), dim=(1, 2, 3))

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its lambda '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self):
        # NOTE -kld actually
        return 0.5 * torch.sum(
                1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
            dim=(1)
        )

    def loss_function(self, recon_x, x):
        rec_term = self.xent_continuous_ber(recon_x, x)
        rec_term = torch.mean(rec_term)

        kld = torch.mean(self.kld())

        L = (rec_term + self.beta * kld)

        loss = L

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            '-beta*kld': self.beta * kld
        }

        return loss, loss_dict

    def step(self, input_mb):
        recon_mb, _ = self.forward(input_mb)

        loss, loss_dict = self.loss_function(recon_mb, input_mb)

        recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict

    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

        
