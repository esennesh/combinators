#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import LogNormal, MultivariateNormal, Normal
from torch.distributions.transforms import LowerCholeskyTransform
from torch.nn.functional import softplus

import combinators.model

class InitBallDynamics(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('uncertainty__loc', torch.ones(2))
        self.register_buffer('uncertainty__scale', torch.ones(2))

        self.register_buffer('noise__loc', torch.ones(2))
        self.register_buffer('noise__scale', torch.ones(2))

    def forward(self, p, batch_shape=(1,)):
        loc = self.uncertainty__loc.expand(*batch_shape, 2)
        scale = self.uncertainty__scale.expand(*batch_shape, 2)
        uncertainty = softplus(p.normal(loc, scale, name='uncertainty'))

        loc = self.noise__loc.expand(*batch_shape, 2)
        scale = self.noise__scale.expand(*batch_shape, 2)
        noise = softplus(p.normal(loc, scale, name='noise'))
        return uncertainty, noise

class InitialBallState(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer('velocity_0__loc', torch.ones(2) / np.sqrt(2))
        self.register_buffer('velocity_0__scale', torch.ones(2))

        self.register_buffer('position_0__loc', torch.ones(2))
        self.register_buffer('position_0__scale', torch.ones(2))

    def forward(self, p, batch_shape=(1,)):
        loc = self.velocity_0__loc.expand(*batch_shape, 2)
        scale = self.velocity_0__scale.expand(*batch_shape, 2)
        direction = p.normal(loc, scale, name='velocity_0')
        speed = torch.sqrt(torch.sum(direction**2, dim=1))
        direction = direction / speed.unsqueeze(-1).expand(*direction.shape)

        loc = self.position_0__loc.expand(*batch_shape, 2)
        scale = self.position_0__scale.expand(*batch_shape, 2)
        position = p.normal(loc, scale, name='position_0')

        return direction, position

def reflect_on_boundary(position, direction, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = position[:, d] - sign * boundary
    overage = torch.where(torch.sign(overage) == sign, overage,
                          torch.zeros(*overage.shape, device=position.device))
    position = list(torch.unbind(position, 1))
    position[d] = position[d] - 2 * overage
    position = torch.stack(position, dim=1)

    direction = list(torch.unbind(direction, 1))
    direction[d] = torch.where(overage != 0.0, -direction[d], direction[d])
    direction = torch.stack(direction, dim=1)
    return position, direction

def simulate_step(position, velocity):
    proposal = position + velocity
    for i in range(2):
        for pos in [True, False]:
            proposal, velocity = reflect_on_boundary(
                proposal, velocity, 6.0, d=i, positive=pos
            )
    return proposal, velocity

def simulate_trajectory(position, velocity, num_steps, velocities=None):
    trajectory = torch.zeros(position.shape[0], num_steps + 1, 2, 2)
    trajectory[:, 0, 0] = position
    trajectory[:, 0, 1] = velocity
    for t in range(1, num_steps + 1):
        if velocities is not None:
            velocity = velocities[:, t-1]
        position, velocity = simulate_step(position, velocity)
        trajectory[:, t, 0] = position
        trajectory[:, t, 1] = velocity
    return trajectory

class StepBallDynamics(nn.Module):
    def forward(self, p, direction, position, uncertainty, noise, data={}):
        position, direction = simulate_step(position, direction)

        position = p.normal(loc=position, scale=noise, name='position',
                            value=data['position'])
        direction = p.normal(loc=direction, scale=uncertainty, name='velocity')

        return direction, position

class StepBallProposal(nn.Module):
    def __init__(self):
        super().__init__()

        self.direction_gibbs = nn.Sequential(
            nn.Linear(2 * 6, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 4)
        )

        self.direction_feedback = nn.Sequential(
            nn.Linear(2 * 4, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 2 * 4)
        )

    def forward(self, q, prev_dir, prev_pos, uncertainty, noise, next_dir,
                next_pos, data={}):
        velocity_stats = self.direction_gibbs(torch.cat(
            (prev_dir, prev_pos, next_dir, next_pos, uncertainty, noise),
            dim=1,
        )).view(-1, 2, 2)
        velocity_loc = velocity_stats[:, :, 0]
        velocity_scale = softplus(velocity_stats[:, :, 1])
        q.normal(loc=velocity_loc, scale=velocity_scale, name='velocity')

    def feedback(self, p, prev_dir, prev_pos, uncertainty, noise, data={}):
        stats = self.direction_feedback(torch.cat(
            (p['velocity'].value, p['position'].value, uncertainty, noise),
            dim=1,
        )).view(-1, 2, 4)
        return torch.unbind(stats, dim=-1)
