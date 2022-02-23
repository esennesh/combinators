#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, img_side, glimpse_side, what_dim, hidden_dim):
        super().__init__()

        self._glimpse_side = glimpse_side
        self._img_side = img_side

        self.glimpse_to_img_translate = (self._img_side - self._glimpse_side) /\
                                        self._glimpse_side
        self.img_to_glimpse_translate = (self._img_side - self._glimpse_side) /\
                                        self._img_side

        scale = self._img_side / self._glimpse_side
        self.register_buffer('glimpse_to_img_scale', torch.eye(2) * scale)
        scale = self._glimpse_side / self._img_side
        self.register_buffer('img_to_glimpse_scale', torch.eye(2) * scale)

        self.obj_avg = nn.Sequential(
            nn.Linear(what_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.glimpse_side ** 2)
        )

    @property
    def glimpse_side(self):
        return self._glimpse_side

    @property
    def img_side(self):
        return self._img_side

    def predict_obj_mean(self, whats, detach=True):
        P, B, K, _ = whats.shape
        obj_avgs = self.obj_avg(whats).view(P, B, K, self.glimpse_side,
                                            self.glimpse_side)
        if detach:
            obj_avgs = obj_avgs.detach()
        return obj_avgs

    def reconstruct(self, obj_avgs, wheres):
        reconstructions = self.glimpse2image(obj_avgs, wheres)
        return torch.clamp(reconstructions.sum(dim=2), min=0.0, max=1.0)

    def glimpse2image(self, glimpse, where):
        B, P, K, _ = where.shape
        g2i_scale = self.glimpse_to_img_scale.repeat(*where.shape[:-1], 1, 1)
        g2i_trans = where.unsqueeze(-1) * self.glimpse_to_img_translate
        g2i_trans[:, :, :, 0, :] = -1 * g2i_trans[:, :, :, 0, :]

        g2i = torch.cat((g2i_scale, g2i_trans), dim=-1).view(B * P * K, 2, 3)
        g2i_size = torch.Size([B * P * K, 1, self._img_side, self._img_side])
        grid = F.affine_grid(g2i, g2i_size, align_corners=True)
        img = F.grid_sample(glimpse.view(B * P * K, self._glimpse_side,
                                         self._glimpse_side).unsqueeze(dim=1),
                            grid, mode='nearest', align_corners=True)
        return img.squeeze(dim=1).view(B, P, K, self._img_side, self._img_side)

    def image2glimpse(self, img, where):
        B, P, K, _ = where.shape
        i2g_scale = self.img_to_glimpse_scale.repeat(*where.shape[:-1], 1, 1)
        i2g_trans = where.unsqueeze(-1) * self.img_to_glimpse_translate
        i2g_trans[:, :, :, 1, :] = -1 * i2g_trans[:, :, :, 1, :]

        i2g = torch.cat((i2g_scale, i2g_trans), dim=-1).view(B * P * K, 2, 3)
        i2g_size = torch.Size([B * P * K, 1, self._glimpse_side,
                               self._glimpse_side])
        grid = F.affine_grid(i2g, i2g_size, align_corners=True)
        img = img.unsqueeze(-3).repeat(1, 1, K, 1, 1).view(B * P * K,
                                                           self._img_side,
                                                           self._img_side)
        glimpse = F.grid_sample(img.unsqueeze(dim=1), grid, mode='nearest',
                                align_corners=True)
        return glimpse.squeeze(dim=1).view(B, P, K, self._glimpse_side,
                                           self._glimpse_side)
