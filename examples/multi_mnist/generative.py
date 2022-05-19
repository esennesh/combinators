#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialObjectCodes(nn.Module):
    def __init__(self, what_dim, num_objects):
        super().__init__()
        self._dim = what_dim
        self._num_objects = num_objects

        self.register_buffer('what_loc', torch.zeros(what_dim))
        self.register_buffer('what_scale', torch.ones(what_dim))

    @property
    def num_objects(self):
        return self._num_objects

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        what_locs = self.what_loc.expand(*particle_shape, *batch_shape,
                                         self._num_objects, self._dim)
        what_scales = self.what_scale.expand(*particle_shape, *batch_shape,
                                             self._num_objects, self._dim)

        return p.normal(what_locs, what_scales, name='z^{what}')

class InitialObjectLocations(nn.Module):
    def __init__(self, where_dim, num_objects):
        super().__init__()
        self._dim = where_dim
        self._num_objects = num_objects

        self.register_buffer('where_0_loc', torch.zeros(where_dim))
        self.register_buffer('where_0_scale', torch.ones(where_dim))

    @property
    def num_objects(self):
        return self._num_objects

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        where_locs = self.where_0_loc.expand(*particle_shape, *batch_shape,
                                             self._num_objects, self._dim)
        where_scales = self.where_0_scale.expand(*particle_shape, *batch_shape,
                                                 self._num_objects, self._dim)

        return p.normal(where_locs, where_scales, name='z^{where}')

def img_cross_entropy(value, target):
    losses = F.binary_cross_entropy(value, target, reduction='none')
    return losses.sum(dim=-1).sum(dim=-1)

class StepObjects(nn.Module):
    def __init__(self, where_dim, spatial_transform):
        super().__init__()

        self.add_module('spatial_transformer', spatial_transform)

        self.register_buffer('where_t_scale', torch.ones(where_dim) * 0.2)

    def reconstruct(self, obj_avgs, wheres):
        obj_avgs = obj_avgs.unsqueeze(dim=3)
        wheres = wheres.unsqueeze(dim=3)
        reconstructions = self.spatial_transformer.glimpse2image(obj_avgs,
                                                                 wheres)
        reconstructions = reconstructions.squeeze(dim=3)
        return torch.clamp(reconstructions.sum(dim=2), min=0.0, max=1.0)

    def forward(self, p, wheres, whats, data=None):
        P, B, K, _ = whats.shape
        obj_avgs = self.spatial_transformer.predict_obj_mean(whats, False)
        wheres_t = p.normal(wheres, self.where_t_scale, name='z^{where}')

        reconstructions = self.reconstruct(obj_avgs, wheres_t)
        p.loss(img_cross_entropy, reconstructions, data, name='x')

        return wheres_t
