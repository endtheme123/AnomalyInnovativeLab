import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torchvision.models.resnet import resnet18
from math import exp
from vae import VAE

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class VQVAE(VAE):
    
    def __init__(self, img_size, nb_channels, latent_img_size, z_dim,
        rec_loss="xent", beta=1, delta=1, gamma=1, dist="mse",
        num_embed=512, dataset="mvtec"):
        # NOTE all the parameters from VAE are still existing and taking memory
        # even thouhgh not used

        super().__init__(img_size, nb_channels, latent_img_size, z_dim, rec_loss,
            beta, delta)
        self.gamma = gamma
        self.dist = dist
        self.codebook = VQEmbedding(num_embed, z_dim)
        self.num_embed = num_embed

        input_dim = nb_channels
        dim = z_dim

        # overrides VAE final_encoder because there we need to double z_dim
        # NOTE that it is not used if we used Oord encoder
        self.final_encoder = nn.Sequential(
            nn.Conv2d(self.max_depth_conv, self.z_dim, kernel_size=1, stride=1,
            padding=0)
        )

        if dataset in ["mvtec", "got", "semmacape", "kelonia", "miad"]:
            self.oord_encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
                ResBlock(dim),
                ResBlock(dim),
                ResBlock(dim),
            )
            self.oord_decoder = nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            #    nn.Tanh()
            )
        if dataset == "UCSDped1" or dataset == "cifar10":
            # Delete one convolutional layer for 32x32 latent space
            self.oord_encoder = nn.Sequential(
                nn.Conv2d(input_dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 4, 2, 1),
                ResBlock(dim),
                ResBlock(dim),
                ResBlock(dim),
            )
            self.oord_decoder = nn.Sequential(
                ResBlock(dim),
                ResBlock(dim),
                ResBlock(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 4, 2, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            )

    def encoder(self, x):

        z_e_x = self.oord_encoder(x)

        return z_e_x

    def decoder(self, z_q_x_st):
        ''' can also be z_q_x simply but a difference is made is forward for
            the gradient to flow '''
        x = self.oord_decoder(z_q_x_st)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        z_e_x = self.encoder(x)
        self.z_e_x = z_e_x
        
        z = self.codebook(z_e_x) # this is where gradient cannot flow
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        self.z_q_x = z_q_x
        x = self.decoder(z_q_x_st)


        return x, (z_e_x, z, z_q_x)

    def generate(self, nb_samples):
        z = torch.randint(
            low=0,
            high=self.num_embed,
            size=(nb_samples, self.latent_img_size, self.latent_img_size)
        )
        z_q_x = self.codebook.embedding(z).permute(0, 3, 1, 2)
        return self.decoder(z_q_x), z

    def loss_function(self, recon_x, x, z_e_x, z_q_x):
        if self.rec_loss == "xent":
            rec_term = torch.mean(self.xent_continuous_ber(recon_x, x))
        elif self.rec_loss == "mse":
            rec_term = torch.mean(-self.mse(recon_x, x))

        # this trains the codebook this is alignment loss
        # can be seen as a Kmeans clustering where the codebook centroids move
        if self.dist == "mse":
            loss_e = torch.mean(
                torch.sum(
                    nn.functional.mse_loss(
                        z_q_x, 
                        z_e_x.detach(),
                        reduction='none'
                    ),
                    dim=(1, 2, 3)
                ),
                dim=0
            )
        elif self.dist == "cos":
            loss_e = torch.mean(
                nn.functional.cosine_similarity(z_q_x, z_e_x.detach())
            )

        # this trains the inputs embedding this is comitment loss
        # this forces the input embeding to be close comitted to their codebook
        # vector
        if self.dist == "mse":
            loss_c = torch.mean(
                torch.sum(
                    nn.functional.mse_loss(
                        z_e_x, 
                        z_q_x.detach(),
                        reduction='none'
                    ),
                    dim=(1, 2, 3)
                ),
                dim=0
            )
        elif self.dist == "cos":
            loss_c = torch.mean(
                nn.functional.cosine_similarity(z_e_x, z_q_x.detach())
            )

        loss = (rec_term - self.beta * loss_e - self.gamma * loss_c)

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            'beta*loss_e': self.beta * loss_e,
            'gamma*loss_c': self.gamma * loss_c
        }

        return loss, loss_dict


    def step(self, input_mb, anneal_tau=False):
        # move tau (parameter of the codebook) at each 100 step
        recon_mb, (z_e_x, z, z_q_x) = self.forward(input_mb)

        loss, loss_dict = self.loss_function(recon_mb, input_mb,
            z_e_x, z_q_x)

        if self.rec_loss == "xent":
            # NOTE do this after the loss function
            recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict
