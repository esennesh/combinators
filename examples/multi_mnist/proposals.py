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
            nn.Linear(spatial_transform.glimpse_side ** 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.what_loc = nn.Linear(hidden_dim // 2, what_dim)
        self.what_log_scale = nn.Linear(hidden_dim // 2, what_dim)

    def forward(self, q, wheres_fb):
        images = torch.stack(wheres_fb['images'], dim=2)
        wheres = torch.stack(wheres_fb['z_where'], dim=3)
        cropped = self.spatial_transformer.image2glimpse(images, wheres)
        cropped = torch.flatten(cropped, -2, -1)
        hiddens = self.object_hiddens(cropped).mean(dim=3)

        loc = self.what_loc(hiddens)
        scale = self.what_log_scale(hiddens).exp()

        q.normal(loc, scale, name='z^{what}')

    def feedback(self, p):
        return ()

class InitialLocationsProposal(nn.Module):
    def __init__(self, spatial_transform, img_side, glimpse_side, hidden_dim,
                 where_dim):
        super().__init__()
        self.spatial_transformer = spatial_transform
        self._img_side = img_side

        frame_side = img_side  - glimpse_side + 1
        self.coordinate_hiddens = nn.Sequential(
            nn.Linear(frame_side ** 2, hidden_dim), nn.ReLU()
        )
        self.where_loc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim), nn.Tanh()
        )
        self.where_log_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim)
        )

    def forward(self, q, recons, data=None):
        _, _, K, glimpse_side, _ = recons.shape
        P, B, _, img_side, _ = data.shape

        locs = []
        scales = []
        q_wheres = []
        framebuffer = data
        for k in range(K):
            features = framebuffer.view(P * B, img_side, img_side).unsqueeze(0)
            kernel = recons[:, :, k, :, :].view(P * B, glimpse_side,
                                                glimpse_side).unsqueeze(1)
            features = F.conv2d(features, kernel, groups=int(P * B))
            frame_side = features.shape[-1]
            features = F.softmax(features.squeeze(0).view(P, B,
                                                          frame_side ** 2),
                                 dim=-1)

            hiddens = self.coordinate_hiddens(features)
            where_loc = self.where_loc(hiddens)
            where_scale = self.where_log_scale(hiddens).exp()

            locs.append(where_loc)
            scales.append(where_scale)

            where = dist.Normal(where_loc, where_scale).rsample()
            q_wheres.append(where)

            reconstruction = self.spatial_transformer.glimpse2image(
                recons[:, :, k, :, :].unsqueeze(dim=2).unsqueeze(dim=3),
                where.unsqueeze(dim=2).unsqueeze(dim=3)
            ).squeeze(dim=3).squeeze(dim=2)
            framebuffer = framebuffer - reconstruction

        where_loc = torch.stack(locs, dim=2)
        where_scale = torch.stack(scales, dim=2)
        where = torch.stack(q_wheres, dim=2)

        q.normal(where_loc, where_scale, value=where, name='z^{where}')

    def feedback(self, p, data=None):
        return ()

class StepLocationsProposal(nn.Module):
    def __init__(self, spatial_transform, img_side, glimpse_side, hidden_dim,
                 where_dim):
        super().__init__()
        self.spatial_transformer = spatial_transform
        self._img_side = img_side

        frame_side = img_side - glimpse_side + 1
        self.coordinate_hiddens = nn.Sequential(
            nn.Linear(frame_side ** 2, hidden_dim), nn.ReLU()
        )
        self.where_loc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim), nn.Tanh()
        )
        self.where_log_scale = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, where_dim)
        )

    def forward(self, q, wheres, whats, wheres_fb, data=None):
        recons = self.spatial_transformer.predict_obj_mean(whats, True)
        _, _, K, glimpse_side, _ = recons.shape
        P, B, img_side, _ = data.shape

        locs = []
        scales = []
        q_wheres = []
        framebuffer = data
        for k in range(K):
            features = framebuffer.reshape(P * B, img_side, img_side)
            features = features.unsqueeze(0)
            kernel = recons[:, :, k, :, :].view(P * B, glimpse_side,
                                                glimpse_side).unsqueeze(1)
            features = F.conv2d(features, kernel, groups=int(P * B))
            frame_side = features.shape[-1]
            features = F.softmax(features.squeeze(0).view(P, B,
                                                          frame_side ** 2),
                                 dim=-1)

            hiddens = self.coordinate_hiddens(features)
            where_loc = self.where_loc(hiddens)
            where_scale = self.where_log_scale(hiddens).exp()

            locs.append(where_loc)
            scales.append(where_scale)

            where = dist.Normal(where_loc, where_scale).sample()
            q_wheres.append(where)

            reconstruction = self.spatial_transformer.glimpse2image(
                recons[:, :, k, :, :].unsqueeze(dim=2).unsqueeze(dim=3),
                where.unsqueeze(dim=2).unsqueeze(dim=3)
            ).squeeze(dim=3).squeeze(dim=2)
            framebuffer = framebuffer - reconstruction

        where_loc = torch.stack(locs, dim=2)
        where_scale = torch.stack(scales, dim=2)
        where = torch.stack(q_wheres, dim=2)

        q.normal(where_loc, where_scale, value=where, name='z^{where}')

    def feedback(self, p, wheres, whats, data=None):
        recons = self.spatial_transformer.predict_obj_mean(whats, True)
        wheres_fb = {'recons': recons, 'image': data}
        whats_fb = {'images': data, 'z_where': p['z^{where}'].value}
        return (wheres_fb, whats_fb)
