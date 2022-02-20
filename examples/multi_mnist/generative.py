#!/usr/bin/env python3

import torch
import torch.nn as nn

class InitialObjectCodes(nn.Module):
    def __init__(self, what_dim):
        super().__init__()
        self._dim = what_dim

        self.register_buffer('what_loc', torch.zeros(what_dim))
        self.register_buffer('what_scale', torch.ones(what_dim))

    def forward(self, p, batch_shape=(1, 1, 1)):
        what_locs = self.what_loc.expand(*batch_shape, self._dim)
        what_scales = self.what_scale.expand(*batch_shape, self._dim)

        return p.normal(what_locs, what_scales, name='z^{what}')

class InitialObjectLocations(nn.Module):
    def __init__(self, where_dim):
        super().__init__()
        self._dim = where_dim

        self.register_buffer('where_0_loc', torch.zeros(where_dim))
        self.register_buffer('where_0_scale', torch.ones(where_dim))

    def forward(self, p, batch_shape=(1, 1, 1)):
        where_locs = self.where_0_loc.expand(*batch_shape, self._dim)
        where_scales = self.where_0_scale.expand(*batch_shape, self._dim)

        return p.normal(where_locs, where_scales, name='z^{where}')

class StepObjects(nn.Module):
    def __init__(self, hidden_dim, where_dim, what_dim, spatial_transform):
        super().__init__()

        self.add_module('spatial_transformer', spatial_transform)

        self.obj_avg = nn.Sequential(
            nn.Linear(what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.spatial_transformer.img_side ** 2)
        )

        self.register_buffer('where_t_scale', torch.ones(where_dim) * 0.2)

    def predict_obj_mean(self, whats):
        P, B, K, _ = whats.shape
        obj_avgs = self.obj_avg(whats).view(P, B, K,
                                            self.spatial_transformer.img_side,
                                            self.spatial_transformer.img_side)
        return obj_avgs.detach()

    def reconstruct(self, obj_avgs, wheres):
        reconstructions = self.spatial_transformer.glimpse2image(obj_avgs,
                                                                 wheres)
        return torch.clamp(reconstructions.squeeze(dim=2).sum(dim=-3), min=0.0,
                           max=1.0)

    def forward(self, p, wheres, whats, data=None):
        P, B, K, _ = whats.shape
        obj_avgs = self.obj_avg(whats).view(P, B, K,
                                            self.spatial_transformer.img_side,
                                            self.spatial_transformer.img_side)

        wheres_t = p.normal(wheres, self.where_t_scale, name='z^{where}')
        wheres_t = wheres_t.unsqueeze(dim=2)

        reconstructions = self.reconstruct(obj_avgs, wheres_t)
        p.bernoulli(reconstructions, name='x', value=data)

        return wheres
