# -*- coding: utf-8 -*-
# @Time    : ***
# @Author  : ***
# @Email   : ***
# @File    : utils.py
# @Software: ***


import os

import torch
import torch.nn as nn
from torchvision.utils import make_grid


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, nrow):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=nrow), cur_iter)


def load_model(model, ema, pretrained):
    # model.load_state_dict(torch.load(pretrained))
    weights = torch.load(pretrained)

    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if 'ema_shadow' in weights.keys():
        ema.shadow = weights['ema_shadow']


def save_checkpoint(model, ema, epoch, iteration, prefix=""):
    state = {'model': model, 'ema_shadow': ema.shadow}
    model_out_path = "model/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class EMA:
    '''ExponentialMovingAverage'''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                # param.data = self.shadow[name]
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                # param.data = self.backup[name]
                param.data.copy_(self.backup[name])
        self.backup = {}
