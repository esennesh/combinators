#!/usr/bin/env python3

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F

class ObjectCodesProposal(nn.Module):
    def __init__(self, spatial_transform, hidden_dim, what_dim):
        super().__init__()
        self.spatial_transformer = spatial_transform

        self.object_hiddens = nn.Sequential(
            nn.Linear(spatial_transform.img_side ** 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
        )
        self.what_loc = nn.Linear(hidden_dim // 2, what_dim)
        self.what_log_scale = nn.Linear(hidden_dim // 2, what_dim)

    def forward(self, q, wheres, frames):
        cropped = self.spatial_transformer.image2glimpse(frames, wheres)
        cropped = torch.flatten(cropped, -2, -1)
        hiddens = self.object_hiddens(cropped).mean(dim=2)

        loc = self.what_loc(hiddens)
        scale = self.what_log_scale(hiddens).exp()

        q.normal(loc, scale, name='z^{what}')

class StepLocationsProposal(nn.Module):
    def __init__(self, spatial_transform, features_side, hidden_dim, where_dim):
        super().__init__()
        self.spatial_transformer = spatial_transform
        self._features_side = features_side

        self.coordinate_hiddens = nn.Sequential(
            nn.Linear(features_side ** 2, hidden_dim), nn.ReLU()
        )
        self.where_loc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim), nn.Tanh()
        )
        self.where_log_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim)
        )

    def forward(self, q, conv_kernel, data=None):
        _, _, K, glimpse_side, _ = conv_kernel.shape
        P, B, _, img_side, _ = data.shape

        locs = []
        scales = []
        wheres = []
        framebuffer = data
        for k in range(K):
            features = framebuffer.view(P * B, img_side, img_side).unsqueeze(0)
            kernel = conv_kernel[:, :, k, :, :].view(P * B, glimpse_side,
                                                     glimpse_side).unsqueeze(1)
            features = F.conv2d(features, kernel, groups=int(P * B))
            features_side = features.shape[-1]
            features = F.softmax(features.squeeze(0).view(P, B,
                                                          features_side ** 2),
                                 dim=-1)

            hiddens = self.coordinate_hiddens(features)
            where_loc = self.where_loc(hiddens)
            where_scale = self.where_log_scale(hiddens).exp()

            locs.append(where_loc)
            scales.append(where_scale)

            where = dist.Normal(where_loc, where_scale).rsample()
            wheres.append(where)

            reconstruction = self.spatial_transformer.glimpse2image(
                conv_kernel[:, :, k, :, :].unsqueeze(dim=2),
                where.unsqueeze(dim=2).unsqueeze(dim=2)
            ).squeeze(dim=2).squeeze(dim=2)
            framebuffer = framebuffer - reconstruction

        where_loc = torch.cat(locs, dim=2)
        where_scale = torch.cat(scales, dim=2)
        where = torch.cat(wheres, dim=2)

        q.normal(where_loc, where_scale, value=where, name='z^{where}')
