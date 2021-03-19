# -*- coding: utf-8 -*-
# @Time    : ***
# @Author  : ***
# @Email   : ***
# @File    : networks.py
# @Software: ***


import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from torch.autograd import Variable
import torch.nn.functional as F


from CustomLayers import *


class Encoder(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Encoder, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.layers = nn.ModuleList()

        self.hdim = hdim
        cc, sz = channels[0], image_size
        self.layers.append(DisInitialBlock(cdim, out_channels=cc, feature_size=sz))

        for ch in channels[1:]:
            self.layers.append(DisGeneralResBlock(cdim, cc, ch, sz // 2))
            cc, sz = ch, sz // 2
        self.layers.append(DisFinalBlock(cdim, in_channels=cc, out_channels=hdim, feature_size=sz // 2))

    def forward(self, rgb):
        x = rgb
        for block in self.layers[:-1]:
            x = block(x)
        y, logit = self.layers[-1](x)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar, logit


class Generator(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(Generator, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        self.layers = nn.ModuleList()

        cc, sz = channels[-1], 4

        self.layers.append(GenInitialBlock(cdim=cdim, hdim=hdim, cc=cc))
        for ch in channels[::-1]:
            self.layers.append(GenGeneralResBlock(in_channels=cc, out_channels=ch, feature_size=sz, cdim=cdim))
            cc, sz = ch, sz * 2
        self.layers.append(GenFinalResBlock(in_channels=cc, feature_size=sz, cdim=cdim))

        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, y):
        out = None
        for block in self.layers:
            y, rgb = block(y)

            if rgb is None:
                out = torch.add(out, y)
                break

            if out is None:
                out = rgb
            else:
                out = torch.add(self.upsampler(out), rgb)
        return torch.sigmoid(out)


class AEGI(nn.Module):
    def __init__(self, cdim=3, hdim=512, channels=[64, 128, 256, 512, 512, 512], image_size=256):
        super(AEGI, self).__init__()

        self.hdim = hdim
        self.encoder = Encoder(cdim, hdim, channels, image_size)
        self.generator = Generator(cdim, hdim, channels, image_size)

    def forward(self, x):
        mu, logvar, logit = self.encode(x)
        z, _ = self.reparameterize(mu, logvar)
        y = self.generate(z)
        return mu, logvar, logit, z, y

    def sample(self, z):
        y = self.generate(z)
        return y

    def encode(self, x):
        mu, logvar, logit = data_parallel(self.encoder, x)
        return mu, logvar, logit

    def generate(self, z):
        y = data_parallel(self.generator, z)
        return y

    def reparameterize(self, mu, logvar, eps=None):
        std = logvar.mul(0.5).exp_()

        if eps is None:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = Variable(eps)

        return eps.mul(std).add_(mu), eps

    def kl_loss(self, mu1, logvar1, mu2=None, logvar2=None):
        if mu2 is None:
            kl = 1 - mu1**2 - logvar1.exp() + logvar1
        else:
            kl = 1 - (mu1 - mu2)**2 / logvar2.exp() - logvar1.exp() / logvar2.exp() + logvar1 - logvar2
        kl = kl.sum(dim=-1) / (-2)
        return kl

    def reconstruction_loss(self, prediction, target, size_average=False):
        error = (prediction - target).view(prediction.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=-1)

        if size_average:
            error = error.mean()
        else:
            error = error.sum()

        return error

    def D_logistic_r1(self, real_scores_out, fake_scores_out, real_images, device, gamma=10.0):
        loss = F.softplus(fake_scores_out)  # -log(1-sigmoid(fake_scores_out))
        loss = loss + F.softplus(-real_scores_out)  # -log(sigmoid(real_scores_out)) # pylint: disable=invalid-unary-operand-type
         # 输出y关于输入x的求导，这里的gradients就是导数
        weight = torch.ones(real_scores_out.size()).to(device)
        real_grads = torch.autograd.grad(
            outputs=real_scores_out,
            inputs=real_images,
            grad_outputs=weight,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = torch.sum(real_grads ** 2, dim=[1, 2, 3])
        reg = gradient_penalty * (gamma * 0.5)

        return loss, reg.view(real_images.shape[0], -1)

    def G_logistic_nonsaturating(self, fake_scores_out):
        loss = F.softplus(-fake_scores_out)
        return loss.mean()

