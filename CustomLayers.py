# -*- coding: utf-8 -*-
# @Time    : ***
# @Author  : ***
# @Email   : ***
# @File    : CustomLayers.py
# @Software: ***


import math
import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import conv2d, conv_transpose2d, linear
from torch.nn.modules.utils import _pair


def exists(val):
    return val is not None


def normal_init(m):
    '''
    initialization
    :param m:
    :return:
    '''
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.normal_(m.weight, mean=0, std=0.002)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            initializer(self._modules[block])

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(nn.Module):
    """ Module implementing the initial block of the Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """
    def __init__(self, cdim=3, hdim=512, cc=512):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use the equalized learning rate
        """
        super().__init__()

        self.main = nn.Sequential()
        self.main.add_module('Init_linear', nn.Linear(hdim, cc * 4 * 4))
        self.main.add_module('Init_lr', nn.LeakyReLU(0.2, True))

        # toRGB
        self.toRGB = ToRGB(cc, cdim, 4)

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, z):
        z = z.view(z.size(0), -1)

        # perform the forward computations:
        y = self.main(z)
        y = y.view(z.size(0), -1, 4, 4)
        rgb = self.toRGB(y)
        return y, rgb


class GenGeneralResBlock(nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, feature_size, cdim):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use the equalized learning rate
        """
        super().__init__()
        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, ch, scale=1.0))

        # toRGB
        self.toRGB = ToRGB(ch, cdim, sz * 2)

    def forward(self, x):
        y = self.main(x)
        rgb = self.toRGB(y)
        return y, rgb


class GenFinalResBlock(nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, feature_size, cdim):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use the equalized learning rate
        """
        super().__init__()
        cc, sz = in_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x):
        y = self.main(x)
        return y, None


class ToRGB(nn.Module):
    '''feature map to RGB'''
    def __init__(self, in_channels, cdim, feature_size):
        super().__init__()

        cc, sz = in_channels, feature_size
        self.main = nn.Sequential()
        self.main.add_module('toRGB_in_{}'.format(sz), nn.Conv2d(cc, cdim, 1, 1, 0))

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x):
        y = self.main(x)
        return y


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """
    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


class DisInitialBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, cdim, out_channels, feature_size):
        super().__init__()
        cc, sz = out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('conv_in_{}'.format(sz), nn.Conv2d(cdim, cc, 5, 1, 2, bias=False))
        self.main.add_module('bn_in_{}'.format(sz), nn.BatchNorm2d(cc))
        self.main.add_module('lr_in_{}'.format(sz), nn.LeakyReLU(0.2, True))
        self.main.add_module('pool_in_{}'.format(sz), nn.AvgPool2d(2))

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x):
        x = self.main(x)
        return x


class DisGeneralResBlock(nn.Module):
    """ General block in the discriminator  """
    def __init__(self, cdim, in_channels, out_channels, feature_size):
        super().__init__()

        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('res_in_{}'.format(sz), Residual_Block(cc, ch, scale=1.0))
        self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))

        # downsampler
        self.downsampler = nn.Conv2d(cc, ch, 1, stride=2, bias=False)

        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        initializer(self._modules['downsampler'])

    def forward(self, x):
        res = self.main(x)
        x = self.downsampler(x)
        return self.relu(self.bn(res + x))


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, cdim, in_channels, out_channels, feature_size):
        super().__init__()
        cc, ch, sz = in_channels, out_channels, feature_size

        self.main = nn.Sequential()
        self.main.add_module('conv_in_{}'.format(sz), nn.Conv2d(cc+1, cc, 3, 1, 1))
        self.to_encode = nn.Linear(cc * 4 * 4, 2 * ch)
        self.to_logit = nn.Linear(2 * ch, 1)

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        initializer(self._modules['to_encode'])
        initializer(self._modules['to_logit'])
        for block in self._modules['main']:
            initializer(block)

    def forward(self, x):
        y = self.batch_discriminator(x)
        y = self.main(y)
        y = y.view(y.size(0), -1)

        y = self.to_encode(y)
        logit = self.to_logit(y)

        return y, logit


class FromRGB(nn.Module):
    '''feature map to RGB'''
    def __init__(self, cdim, out_channels, feature_size):
        super().__init__()

        ch, sz = out_channels, feature_size
        self.main = nn.Sequential()
        self.main.add_module('FromRGB_conv_in_{}'.format(sz), nn.Conv2d(cdim, ch, 1, 1, 0))

        self.__weight_init()

    def __weight_init(self, mode='normal'):
        if mode == 'normal':
            initializer = normal_init
        for block in self._modules['main']:
            initializer(block)

    def forward(self, rgb):
        y = self.main(rgb)
        return y
