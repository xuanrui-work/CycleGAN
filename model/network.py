from .block import *
from .loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        hidden_dims=(64, 128, 256),
        max_pools=(2, 2, 2),
        hparams=None
    ):
        super().__init__()
        self.hparams = hparams

        layers = []
        output_shape = list(input_shape)

        encoder = CNNEncoder(
            input_shape,
            hidden_dims,
            max_pools
        )
        layers += [encoder]
        output_shape = list(encoder.output_shape)
        
        layers += [nn.Conv2d(output_shape[0], 1, kernel_size=3, padding='same')]
        output_shape[0] = 1

        self.layers = nn.Sequential(*layers)
        self.output_shape = tuple(output_shape)
    
    def forward(self, x):
        return self.layers(x)

class Generator(nn.Module):
    def __init__(
        self,
        input_shape=(3, 32, 32),
        hidden_dims=(64, 128, 128),
        max_pools=(2, 2, 0),
        hparams=None
    ):
        super().__init__()
        self.hparams = hparams

        self.encoder = CNNEncoder(
            input_shape,
            hidden_dims,
            max_pools
        )
        self.decoder = CNNDecoder(
            self.encoder.output_shape,
            hidden_dims[::-1],
            max_pools[::-1],
            input_shape
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CycleGAN(nn.Module):
    def __init__(
        self,
        hparams=None
    ):
        super().__init__()

        self.G_AB = Generator(hparams=hparams)
        self.G_BA = Generator(hparams=hparams)
        self.D_A = PatchDiscriminator(hparams=hparams)
        self.D_B = PatchDiscriminator(hparams=hparams)

        self.set_hparams(hparams)

        # loss functions
        self.criter_gan = GANLoss('lsgan')
        self.criter_cyc = nn.L1Loss()

        # loss dict
        self.loss_dict = {}

        # optimizers
        self.optim_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=hparams['lr'],
            betas=(hparams['betas'][0], hparams['betas'][1])
        )
        self.optim_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=hparams['lr'],
            betas=(hparams['betas'][0], hparams['betas'][1])
        )
    
    def set_hparams(self, hparams):
        if 'cyc' in hparams:
            hparams['cyc_ABA'] = hparams['cyc']
            hparams['cyc_BAB'] = hparams['cyc']
        self.hparams = hparams

        self.G_AB.hparams = hparams
        self.G_BA.hparams = hparams
        self.D_A.hparams = hparams
        self.D_B.hparams = hparams
    
    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    def forward(self, x_A=None, x_B=None):
        assert x_A is not None or x_B is not None, (
            f'either x_A or x_B must be provided, but got x_A={x_A} and x_B={x_B}'
        )

        if x_A is not None:
            fake_AB = self.G_AB(x_A)
            fake_ABA = self.G_BA(fake_AB)
        else:
            fake_AB = fake_ABA = None
        if x_B is not None:
            fake_BA = self.G_BA(x_B)
            fake_BAB = self.G_AB(fake_BA)
        else:
            fake_BA = fake_BAB = None

        outputs = {
            'real_A': x_A,
            'real_B': x_B,
            'fake_AB': fake_AB,
            'fake_BA': fake_BA,
            'fake_ABA': fake_ABA,
            'fake_BAB': fake_BAB
        }
        return outputs

    def backward_D(self, D, real, fake):
        # real
        p_real = D(real)
        l_real = self.criter_gan(p_real, True)
        # fake
        p_fake = D(fake.detach())
        l_fake = self.criter_gan(p_fake, False)
        # total loss
        loss_D = 0.5 * (l_real + l_fake)
        return loss_D

    def backward_D_A(self, outputs):
        fake_A = outputs['fake_BA']
        l_D_A = self.backward_D(self.D_A, outputs['real_A'], fake_A)
        self.loss_dict['l_D_A'] = l_D_A
        return l_D_A

    def backward_D_B(self, outputs):
        fake_B = outputs['fake_AB']
        l_D_B = self.backward_D(self.D_B, outputs['real_B'], fake_B)
        self.loss_dict['l_D_B'] = l_D_B
        return l_D_B

    def backward_G(self, outputs):
        # GAN loss D_B(G_AB(A))
        l_G_AB = self.criter_gan(self.D_B(outputs['fake_AB']), True)
        # GAN loss D_A(G_BA(B))
        l_G_BA = self.criter_gan(self.D_A(outputs['fake_BA']), True)
        # || G_BA(G_AB(A)) - A ||
        l_cyc_ABA = self.criter_cyc(outputs['fake_ABA'], outputs['real_A'])
        # || G_AB(G_BA(B)) - B ||
        l_cyc_BAB = self.criter_cyc(outputs['fake_BAB'], outputs['real_B'])
        # total loss
        loss_G = (
            l_G_AB +
            l_G_BA +
            self.hparams['cyc_ABA'] * l_cyc_ABA +
            self.hparams['cyc_BAB'] * l_cyc_BAB
        )

        self.loss_dict.update({
            'l_G_AB': l_G_AB,
            'l_G_BA': l_G_BA,
            'l_cyc_A': l_cyc_ABA,
            'l_cyc_B': l_cyc_BAB
        })
        return loss_G
    
    def optimize_params(self, x_A, x_B, backward=True):
        # forward pass
        outputs = self.forward(x_A, x_B)

        # update D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        l_D_A = self.backward_D_A(outputs)
        l_D_B = self.backward_D_B(outputs)
        if backward:
            self.optim_D.zero_grad()
            l_D_A.backward()
            l_D_B.backward()
            self.optim_D.step()

        # update G_A and G_B
        self.set_requires_grad([self.D_A, self.D_B], False)
        loss_G = self.backward_G(outputs)
        if backward:
            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()

        return (outputs, self.loss_dict)
