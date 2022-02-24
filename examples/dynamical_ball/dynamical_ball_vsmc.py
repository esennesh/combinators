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

        self.register_parameter('uncertainty__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('uncertainty__scale',
                                nn.Parameter(torch.ones(2)))

        self.register_parameter('noise__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('noise__scale', nn.Parameter(torch.ones(2)))

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        loc = self.uncertainty__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.uncertainty__scale.expand(*particle_shape, *batch_shape, 2)
        uncertainty = softplus(p.normal(loc, scale, name='uncertainty'))

        loc = self.noise__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.noise__scale.expand(*particle_shape, *batch_shape, 2)
        noise = softplus(p.normal(loc, scale, name='noise'))
        return uncertainty, noise

class InitDynamicsProposal(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_parameter('uncertainty__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('uncertainty__scale',
                                nn.Parameter(torch.ones(2)))

        self.register_parameter('noise__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('noise__scale', nn.Parameter(torch.ones(2)))

    def forward(self, q, batch_shape=(1,), particle_shape=(1,), data={}):
        loc = self.uncertainty__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.uncertainty__scale.expand(*particle_shape, *batch_shape, 2)
        uncertainty = softplus(q.normal(loc, scale, name='uncertainty'))

        loc = self.noise__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.noise__scale.expand(*particle_shape, *batch_shape, 2)
        noise = softplus(q.normal(loc, scale, name='noise'))

    def feedback(self, p, *args, data={}):
        return ()

class InitialBallState(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_parameter('velocity_0__loc',
                                nn.Parameter(torch.ones(2) / np.sqrt(2)))
        self.register_parameter('velocity_0__scale',
                                nn.Parameter(torch.ones(2)))

        self.register_parameter('position_0__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('position_0__scale',
                                nn.Parameter(torch.ones(2)))

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        loc = self.velocity_0__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.velocity_0__scale.expand(*particle_shape, *batch_shape, 2)
        direction = p.normal(loc, scale, name='velocity_0')
        speed = torch.sqrt(torch.sum(direction**2, dim=2))
        direction = direction / speed.unsqueeze(-1).expand(*direction.shape)

        loc = self.position_0__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.position_0__scale.expand(*particle_shape, *batch_shape, 2)
        position = p.normal(loc, scale, name='position_0')

        return direction, position

class InitBallProposal(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_parameter('velocity_0__loc',
                                nn.Parameter(torch.ones(2) / np.sqrt(2)))
        self.register_parameter('velocity_0__scale',
                                nn.Parameter(torch.ones(2)))

        self.register_parameter('position_0__loc', nn.Parameter(torch.ones(2)))
        self.register_parameter('position_0__scale',
                                nn.Parameter(torch.ones(2)))

    def forward(self, q, batch_shape=(1,), particle_shape=(1,), data={}):
        loc = self.velocity_0__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.velocity_0__scale.expand(*particle_shape, *batch_shape, 2)
        direction = q.normal(loc, scale, name='velocity_0')
        speed = torch.sqrt(torch.sum(direction**2, dim=2))
        direction = direction / speed.unsqueeze(-1).expand(*direction.shape)

        loc = self.position_0__loc.expand(*particle_shape, *batch_shape, 2)
        scale = self.position_0__scale.expand(*particle_shape, *batch_shape, 2)
        position = q.normal(loc, scale, name='position_0')

    def feedback(self, p, *args, data={}):
        return ()

def reflect_on_boundary(position, direction, boundary, d=0, positive=True):
    sign = 1.0 if positive else -1.0
    overage = position[:, :, d] - sign * boundary
    overage = torch.where(torch.sign(overage) == sign, overage,
                          torch.zeros(*overage.shape, device=position.device))
    position = list(torch.unbind(position, dim=2))
    position[d] = position[d] - 2 * overage
    position = torch.stack(position, dim=2)

    direction = list(torch.unbind(direction, dim=2))
    direction[d] = torch.where(overage != 0.0, -direction[d], direction[d])
    direction = torch.stack(direction, dim=2)
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
    trajectory = torch.zeros(*position.shape[:2], num_steps + 1, 2, 2)
    trajectory[:, :, 0, 0] = position
    trajectory[:, :, 0, 1] = velocity
    for t in range(1, num_steps + 1):
        if velocities is not None:
            velocity = velocities[:, :, t-1]
        position, velocity = simulate_step(position, velocity)
        trajectory[:, :, t, 0] = position
        trajectory[:, :, t, 1] = velocity
    return trajectory

class StepBallState(nn.Module):
    def forward(self, p, direction, position, uncertainty, noise, data=None):
        position, direction = simulate_step(position, direction)

        position = p.normal(loc=position, scale=noise, name='position',
                            value=data)
        direction = p.normal(loc=direction, scale=uncertainty, name='velocity')

        return direction, position

class StepBallProposal(nn.Module):
    def __init__(self):
        super().__init__()

        self.velocity_loc = nn.Sequential(
            nn.Linear(2 * 5, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 2)
        )

        self.velocity_log_scale = nn.Sequential(
            nn.Linear(2 * 5, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, q, prev_dir, prev_pos, uncertainty, noise, data={}):
        velocity_stats = torch.cat(
            (prev_dir, prev_pos, uncertainty, noise, data),
            dim=-1,
        )
        
        velocity_loc = self.velocity_loc(velocity_stats)
        velocity_scale = self.velocity_log_scale(velocity_stats).exp()
        
        q.normal(loc=velocity_loc, scale=velocity_scale, name='velocity')

    def feedback(self, p, prev_dir, prev_pos, uncertainty, noise, data={}):
        stats = self.direction_feedback(torch.cat(
            (p['velocity'].value, p['position'].value, uncertainty, noise),
            dim=2,
        )).view(-1, noise.shape[1], 2, 4)
        return torch.unbind(stats, dim=-1)
