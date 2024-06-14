import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import warnings

class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (3, 32, 32),
        hidden_dims: list[int] = None,
        max_pools: list[int] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.input_shape = input_shape
        self.hidden_dims = hidden_dims
        self.max_pools = max_pools
        
        conv_layers = []
        output_shape = list(input_shape)

        for i, hidden_dim in enumerate(hidden_dims):
            conv_layers += [nn.Conv2d(
                output_shape[0],
                hidden_dim,
                kernel_size=3,
                padding='same'
            )]
            if max_pools[i]:
                conv_layers += [nn.MaxPool2d(kernel_size=max_pools[i])]
                output_shape[1] //= max_pools[i]
                output_shape[2] //= max_pools[i]

            conv_layers += [nn.BatchNorm2d(hidden_dim)]
            conv_layers += [nn.LeakyReLU(0.2, inplace=True)]
            output_shape[0] = hidden_dim
        
        output_shape = tuple(output_shape)

        self.conv_layers = nn.Sequential(*conv_layers)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.conv_layers(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dims: list[int] = None,
        upsamples: list[int] = None,
        output_shape: tuple[int, int, int] = (3, 32, 32),
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512][::-1]

        self.input_shape = input_shape

        conv_layers = []
        output_shape1 = list(input_shape)

        for i, hidden_dim in enumerate(hidden_dims):
            if upsamples[i]:
                conv_layers += [nn.Upsample(scale_factor=upsamples[i], mode='bilinear')]
                output_shape1[1] *= upsamples[i]
                output_shape1[2] *= upsamples[i]
            conv_layers += [nn.Conv2d(
                output_shape1[0],
                hidden_dim,
                kernel_size=3,
                padding='same'
            )]

            conv_layers += [nn.BatchNorm2d(hidden_dim)]
            conv_layers += [nn.LeakyReLU(0.2, inplace=True)]
            output_shape1[0] = hidden_dim
        
        output_shape1 = tuple(output_shape1)
        
        self.conv_layers = nn.Sequential(*conv_layers)

        output_layer = [
            nn.Conv2d(hidden_dims[-1], output_shape[0], kernel_size=3, padding='same'),
            nn.Sigmoid()
        ]
        if output_shape1[1:] != output_shape[1:]:
            output_layer.insert(
                0,
                nn.Upsample(size=output_shape[1:], mode='bilinear')
            )
            warnings.warn(
                f'Mismatch between model output shape output_shape1={output_shape1} and output_shape={output_shape}. '
                f'A nn.Upsample has been prepended to the output layer to give the desired output_shape.'
            )
        
        self.output_layer = nn.Sequential(*output_layer)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.output_layer(x)
        return x
