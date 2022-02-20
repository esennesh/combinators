#!/usr/bin/env python3

import torch
import torch.nn as nn

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
