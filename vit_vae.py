
import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18

from transformers import ViTModel
import torch.nn.functional as F
import copy

class ViTVAE(nn.Module):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
        '''
        '''
        super(ViTVAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta
        self.rec_loss = rec_loss
        self.delta = delta
        # self.nb_conv = int(np.log2((img_size+32) // latent_img_size))
        self.nb_conv = 3
        self.max_depth_conv = 384
        # self.init_en = nn.Sequential(
        #     nn.Conv2d(4, 256,
        #         kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     # nn.ConvTranspose2d(32, 128,
        #     #     kernel_size=5, stride=1, padding=0),
        #     # nn.BatchNorm2d(128),
        #     # nn.ReLU(),
        #     # nn.ConvTranspose2d(32, 256,
        #     #     kernel_size=5, stride=1, padding=0),
        #     # nn.BatchNorm2d(256),
        #     # nn.ReLU(),
        #     nn.ConvTranspose2d(256, 512,
        #         kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )

        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # for param in self.vit.parameters():
        #     param.requires_grad = False
        # self.vit = ViT(
        #                 image_size = 256,
        #                 patch_size = 16,
        #                 num_classes = 512,
        #                 dim = 1024,
        #                 depth = 6,
        #                 heads = 16,
        #                 mlp_dim = 2048,
        #                 dropout = 0.0,
        #                 emb_dropout = 0.1
        # )

        # self.final_encoder = nn.Sequential(
        #     nn.Linear(self.vit.config.hidden_size, latent_img_size * latent_img_size),
            
        #     nn.ReLU()
        # )

        self.out_en = nn.Sequential(
            # nn.ConvTranspose2d(1, 256,
            #     kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 128,
            #     kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, 256,
            #     kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Conv2d(3, 512,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        # self.initial_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
        #         kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(self.max_depth_conv),
        #     nn.LeakyReLU()
        # )

        # nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        # self.decoder_layers.append(nn.Sequential(
        #             nn.ConvTranspose2d(256, 128, 7, 1, 0),
        #             nn.BatchNorm2d(128),
        #             nn.ReLU()
        #         ))
        # self.decoder_layers.append(nn.Sequential(
        #             nn.ConvTranspose2d(128, 64, 7, 1, 0),
        #             nn.BatchNorm2d(64),
        #             nn.ReLU()
        #         ))

        # self.decoder_layers.append(nn.Sequential(
        #             nn.ConvTranspose2d(, depth_out, 4, 2, 1),
        #             nn.BatchNorm2d(depth_out),
        #             nn.ReLU()
        #         ))
        for i in reversed(range(1, 5)):
            depth_in = 3*2 ** (2+ i + 1)
            depth_out = 3*2 ** (2 + i)
            # print("depth out: ", depth_out)
            if i == 1:
                depth_out = self.nb_channels
                self.decoder_layers.append(nn.Sequential(

                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                ))
            else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                ))
        self.conv_decoder = nn.Sequential(
            *self.decoder_layers
        )

    def encoder(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.vit(x)[0][:,1:,:]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], 14, 14)
        print("vit output: ", x.shape)
        # x = self.final_encoder(x)
        # print("final encoder output: ", x.shape)
        
        # batch_size = x.shape[0]
        # x = x.reshape(batch_size,1, 32, 32)
        # x = self.out_en(x)
        # print("final encoder output: ", x.shape)
        return x[:, :self.z_dim], x[:, self.z_dim :]

    def reparameterize(self, mu, logvar):
        if self.training:
          std = torch.exp(torch.mul(logvar, 0.5))
          eps = torch.randn_like(std)
          return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        # z = z.view(z.size(0), -1)
        # z = self.initial_decoder(z)
        x = self.conv_decoder(z)

        # print("decoder output shape: ", x.shape)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        self.logvar = logvar
        return self.decoder(z), (mu, logvar)
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


        





# import numpy as np
# import torch
# from torch import nn
# from torchvision.models.resnet import resnet18

# from transformers import ViTModel
# import torch.nn.functional as F
# import copy

# # class ViT_VAE(nn.Module):

# #     def __init__(self,batch_size, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
# #         '''
# #         '''
# #         super(ViT_VAE, self).__init__()

# #         self.img_size = img_size
# #         self.nb_channels = nb_channels
# #         self.latent_img_size = latent_img_size
# #         self.z_dim = z_dim
# #         self.beta = beta
# #         self.rec_loss = rec_loss
# #         self.delta = delta
# #         self.dropout = nn.Dropout(p=0.5)
# #         self.batch_size = batch_size

# #         # [b, 784] => [b, 20]
# #         # u: [b, 10]
# #         # sigma: [b, 10]
# #         self.vit = ViT(
# #                         image_size = 256,
# #                         patch_size = 16,
# #                         num_classes = 256,
# #                         dim = 1024,
# #                         depth = 6,
# #                         heads = 16,
# #                         mlp_dim = 2048,
# #                         dropout = 0.0,
# #                         emb_dropout = 0.1
# #         )
# #         print(self.vit)

# #         self.linear_latent_channel  = nn.Sequential(
# #              nn.Linear(256, 512),
# #             #  nn.Dropout(0.5),
# #              nn.GELU(),
# #              nn.Linear(512, 1024),
# #             #  nn.Dropout(0.5),
# #              nn.GELU(),

# #         )

# #         # self.linear_stack = []
# #         # for i in range(32):
# #         #     self.linear_stack.append(copy.deepcopy(self.linear_latent_channel))
        
# #         # self.linear_up = nn.Sequential(
# #         #     nn.Linear(16, 128),
# #         #     nn.BatchNorm1d(128),
# #         #     nn.GELU(),
# #         #     nn.Linear(128, 256),
# #         #     nn.BatchNorm1d(256),
# #         #     nn.GELU(),
# #         #     nn.Linear(256, 3*224*224),

# #         # )
# #         self.initial_decoder = nn.Sequential(
# #             nn.ConvTranspose2d(1, 512,
# #                 kernel_size=1, stride=1, padding=0),
# #             nn.BatchNorm2d(512),
# #             nn.ReLU()
# #         )
# #         self.decoder_layers = []
# #         for i in reversed(range(1,4)):
# #             depth_in = 2 ** (i+1)
# #             depth_out = 2 ** (i+1-1)
# #             if i == 1:
# #                 depth_out = self.nb_channels
# #                 self.decoder_layers.append(nn.Sequential(
# #                     nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
# #                 ))
# #             else:
# #                 self.decoder_layers.append(nn.Sequential(
# #                     nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
# #                     nn.BatchNorm2d(depth_out),
# #                     nn.ReLU()
# #                 ))
# #         self.conv_decoder = nn.Sequential(
# #             *self.decoder_layers
# #         )

# #     # def latent_stablizer(self, x):
# #     #     a = []
# #     #     for layer in self.linear_stack:
# #     #         layer.to(x.device)
            
# #     #         k = layer(x)
            
# #     #         a.append(k.reshape(self.batch_size, 32,32))

# #     #     x = torch.stack(a, dim = 1)

# #     #     return x


# #     def encoder(self, x):
# #         x = self.vit(x)
# #         x = self.linear_latent_channel(x)
# #         x = x.reshape((self.batch_size, 32,32))
# #         x = torch.unsqueeze(x, dim=1)
# #         # x = self.latent_stablizer(x)
# #         x = self.initial_decoder(x)
# #         return x[:, :256], x[:, 256:]

# #     def reparameterize(self, mu, logvar):
# #         if self.training:
# #             std = torch.exp(torch.mul(logvar, 0.5))
# #             eps = torch.randn_like(std)
# #             return eps * std + mu
# #         else:
# #             return mu

# #     def decoder(self, z):
# #         # z = z.reshape(self.batch_size, 32,32)
        
# #         x = self.conv_decoder(z)
        
# #         x = nn.Sigmoid()(x)
# #         return x

# #     def forward(self, x):
# #         mu, logvar = self.encoder(x)
# #         z = self.reparameterize(mu, logvar)
# #         self.mu = mu
# #         print(mu.shape)
# #         self.logvar = logvar
# #         x_rec = self.decoder(z)
# #         # x_rec = x_rec.reshape(8,3,224,224)
# #         return x_rec, (mu, logvar)

# #     def xent_continuous_ber(self, recon_x, x, pixelwise=False):
# #         ''' p(x_i|z_i) a continuous bernoulli '''
# #         eps = 1e-6
# #         def log_norm_const(x):
# #             # numerically stable computation
# #             x = torch.clamp(x, eps, 1 - eps)
# #             x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
# #                     torch.ones_like(x))
# #             return torch.log((2 * self.tarctanh(1 - 2 * x)) /
# #                             (1 - 2 * x) + eps)
# #         if pixelwise:
# #             return (x * torch.log(recon_x + eps) +
# #                             (1 - x) * torch.log(1 - recon_x + eps) +
# #                             log_norm_const(recon_x))
# #         else:
# #             return torch.sum(x * torch.log(recon_x + eps) +
# #                             (1 - x) * torch.log(1 - recon_x + eps) +
# #                             log_norm_const(recon_x), dim=(1, 2, 3))

# #     def mean_from_lambda(self, l):
# #         ''' because the mean of a continuous bernoulli is not its lambda '''
# #         l = torch.clamp(l, 10e-6, 1 - 10e-6)
# #         l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
# #             torch.ones_like(l))
# #         return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

# #     def kld(self):
# #         # NOTE -kld actually
# #         return 0.5 * torch.sum(
# #                 1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
# #             dim=(1)
# #         )

# #     def loss_function(self, recon_x, x):
# #         rec_term = self.xent_continuous_ber(recon_x, x)
# #         rec_term = torch.mean(rec_term)

# #         kld = torch.mean(self.kld())

# #         L = (rec_term + self.beta * kld)

# #         loss = L

# #         loss_dict = {
# #             'loss': loss,
# #             'rec_term': rec_term,
# #             '-beta*kld': self.beta * kld
# #         }

# #         return loss, loss_dict

# #     def step(self, input_mb):
# #         recon_mb, _ = self.forward(input_mb)

# #         loss, loss_dict = self.loss_function(recon_mb, input_mb)

# #         recon_mb = self.mean_from_lambda(recon_mb)

# #         return loss, recon_mb, loss_dict

# #     def tarctanh(self, x):
# #         return 0.5 * torch.log((1+x)/(1-x))

# class ViTVAE(nn.Module):

#     def __init__(self, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
#         '''
#         '''
#         super(ViTVAE, self).__init__()

#         self.img_size = img_size
#         self.nb_channels = nb_channels
#         self.latent_img_size = latent_img_size
#         self.z_dim = z_dim
#         self.beta = beta
#         self.rec_loss = rec_loss
#         self.delta = delta
#         # self.nb_conv = int(np.log2((img_size+32) // latent_img_size))
#         self.nb_conv = 3
#         self.max_depth_conv = 2 ** (4 + self.nb_conv+1)
#         # self.init_en = nn.Sequential(
#         #     nn.Conv2d(4, 256,
#         #         kernel_size=4, stride=2, padding=1),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(),
#         #     # nn.ConvTranspose2d(32, 128,
#         #     #     kernel_size=5, stride=1, padding=0),
#         #     # nn.BatchNorm2d(128),
#         #     # nn.ReLU(),
#         #     # nn.ConvTranspose2d(32, 256,
#         #     #     kernel_size=5, stride=1, padding=0),
#         #     # nn.BatchNorm2d(256),
#         #     # nn.ReLU(),
#         #     nn.ConvTranspose2d(256, 512,
#         #         kernel_size=1, stride=1, padding=0),
#         #     nn.BatchNorm2d(512),
#         #     nn.ReLU()
#         # )

#         self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
#         # self.vit = ViT(
#         #                 image_size = 256,
#         #                 patch_size = 16,
#         #                 num_classes = 512,
#         #                 dim = 1024,
#         #                 depth = 6,
#         #                 heads = 16,
#         #                 mlp_dim = 2048,
#         #                 dropout = 0.0,
#         #                 emb_dropout = 0.1
#         # )

#         # self.final_encoder = nn.Sequential(
#         #     nn.Linear(self.vit.config.hidden_size, latent_img_size * latent_img_size),
            
#         #     nn.ReLU()
#         # )

#         self.out_en = nn.Sequential(
#             # nn.ConvTranspose2d(1, 256,
#             #     kernel_size=4, stride=2, padding=1),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(32, 128,
#             #     kernel_size=5, stride=1, padding=0),
#             # nn.BatchNorm2d(128),
#             # nn.ReLU(),
#             # nn.ConvTranspose2d(32, 256,
#             #     kernel_size=5, stride=1, padding=0),
#             # nn.BatchNorm2d(256),
#             # nn.ReLU(),
#             nn.Conv2d(3, 512,
#                 kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU()
#         )

#         self.initial_decoder = nn.Sequential(
#             nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
#                 kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.max_depth_conv),
#             nn.LeakyReLU()
#         )

#         nb_conv_dec = self.nb_conv

#         self.decoder_layers = []
#         self.decoder_layers.append(nn.Sequential(
#                     nn.ConvTranspose2d(256, 128, 7, 1, 0),
#                     nn.BatchNorm2d(128),
#                     nn.ReLU()
#                 ))
#         self.decoder_layers.append(nn.Sequential(
#                     nn.ConvTranspose2d(128, 64, 7, 1, 0),
#                     nn.BatchNorm2d(64),
#                     nn.ReLU()
#                 ))

#         # self.decoder_layers.append(nn.Sequential(
#         #             nn.ConvTranspose2d(, depth_out, 4, 2, 1),
#         #             nn.BatchNorm2d(depth_out),
#         #             nn.ReLU()
#         #         ))
#         for i in reversed(range(1, nb_conv_dec+1)):
#             depth_in = 2 ** (2 + i + 1)
#             depth_out = 2 ** (2 + i)
#             # print("depth out: ", depth_out)
#             if i == 1:
#                 depth_out = self.nb_channels
#                 self.decoder_layers.append(nn.Sequential(

#                     nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
#                 ))
#             else:
#                 self.decoder_layers.append(nn.Sequential(
#                     nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
#                     nn.BatchNorm2d(depth_out),
#                     nn.ReLU()
#                 ))
#         self.conv_decoder = nn.Sequential(
#             *self.decoder_layers
#         )

#     def encoder(self, x):
#         # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
#         x = self.vit(x)[0][:,0,:]

#         print("vit output: ", x.shape)
#         # x = self.final_encoder(x)
#         print("final encoder output: ", x.shape)
#         x = x.reshape(x.shape[0], 3, 16, 16)
#         # batch_size = x.shape[0]
#         # x = x.reshape(batch_size,1, 32, 32)
#         x = self.out_en(x)
#         print("final encoder output: ", x.shape)
#         return x[:, :self.z_dim], x[:, self.z_dim :]

#     def reparameterize(self, mu, logvar):
#         # if self.training:
#         std = torch.exp(torch.mul(logvar, 0.5))
#         eps = torch.randn_like(std)
#         return eps * std + mu
#         # else:
#         #     return mu

#     def decoder(self, z):
#         # z = z.view(z.size(0), -1)
#         z = self.initial_decoder(z)
#         x = self.conv_decoder(z)

#         # print("decoder output shape: ", x.shape)
#         x = nn.Sigmoid()(x)
#         return x

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         self.mu = mu
#         self.logvar = logvar
#         return self.decoder(z), (mu, logvar)
#     def xent_continuous_ber(self, recon_x, x, pixelwise=False):
#         ''' p(x_i|z_i) a continuous bernoulli '''
#         eps = 1e-6
#         def log_norm_const(x):
#             # numerically stable computation
#             x = torch.clamp(x, eps, 1 - eps)
#             x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
#                     torch.ones_like(x))
#             return torch.log((2 * self.tarctanh(1 - 2 * x)) /
#                             (1 - 2 * x) + eps)
#         if pixelwise:
#             return (x * torch.log(recon_x + eps) +
#                             (1 - x) * torch.log(1 - recon_x + eps) +
#                             log_norm_const(recon_x))
#         else:
#             return torch.sum(x * torch.log(recon_x + eps) +
#                             (1 - x) * torch.log(1 - recon_x + eps) +
#                             log_norm_const(recon_x), dim=(1, 2, 3))

#     def mean_from_lambda(self, l):
#         ''' because the mean of a continuous bernoulli is not its lambda '''
#         l = torch.clamp(l, 10e-6, 1 - 10e-6)
#         l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
#             torch.ones_like(l))
#         return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

#     def kld(self):
#         # NOTE -kld actually
#         return 0.5 * torch.sum(
#                 1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
#             dim=(1)
#         )

#     def loss_function(self, recon_x, x):
#         rec_term = self.xent_continuous_ber(recon_x, x)
#         rec_term = torch.mean(rec_term)

#         kld = torch.mean(self.kld())

#         L = (rec_term + self.beta * kld)

#         loss = L

#         loss_dict = {
#             'loss': loss,
#             'rec_term': rec_term,
#             '-beta*kld': self.beta * kld
#         }

#         return loss, loss_dict

#     def step(self, input_mb):
#         recon_mb, _ = self.forward(input_mb)

#         loss, loss_dict = self.loss_function(recon_mb, input_mb)

#         recon_mb = self.mean_from_lambda(recon_mb)

#         return loss, recon_mb, loss_dict

#     def tarctanh(self, x):
#         return 0.5 * torch.log((1+x)/(1-x))


        
