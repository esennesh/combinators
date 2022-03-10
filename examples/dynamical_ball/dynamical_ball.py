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

        self.uncertainty_gibbs = nn.Sequential(
            nn.Linear(4, 8), nn.PReLU(),
            nn.Linear(8, 8), nn.PReLU(),
            nn.Linear(8, 4)
        )

        self.noise_gibbs = nn.Sequential(
            nn.Linear(4, 8), nn.PReLU(),
            nn.Linear(8, 8), nn.PReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, q, uncertainties, noises, data={}):
        if len(uncertainties.shape) < 3:
            uncertainties = uncertainties.unsqueeze(2)
        if len(noises.shape) < 3:
            noises = noises.unsqueeze(2)
        uncertainty_stats = self.uncertainty_gibbs(torch.cat(
            (uncertainties.mean(dim=2),
             uncertainties.std(dim=2, unbiased=False)), dim=2
        )).view(-1, noises.shape[1], 2, 2).unbind(dim=-1)
        q.normal(uncertainty_stats[0], softplus(uncertainty_stats[1]),
                 name='uncertainty')

        noise_stats = self.noise_gibbs(torch.cat(
            (noises.mean(dim=2), noises.std(dim=2, unbiased=False)), dim=2
        )).view(-1, noises.shape[1], 2, 2).unbind(dim=-1)
        q.normal(noise_stats[0], softplus(noise_stats[1]), name='noise')

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

        self.velocity_gibbs = nn.Sequential(
            nn.Linear(4, 8), nn.PReLU(),
            nn.Linear(8, 8), nn.PReLU(),
            nn.Linear(8, 4)
        )

        self.position_gibbs = nn.Sequential(
            nn.Linear(4, 8), nn.PReLU(),
            nn.Linear(8, 8), nn.PReLU(),
            nn.Linear(8, 4)
        )

    def forward(self, q, direction, position, data={}):
        if len(direction.shape) < 3:
            direction = direction.unsqueeze(1)
        if len(position.shape) < 3:
            position = position.unsqueeze(1)
        stats = torch.cat((direction, position), dim=2)

        vel_stats = self.velocity_gibbs(stats)
        vel_stats = vel_stats.view(-1, direction.shape[1], 2, 2).unbind(dim=2)
        q.normal(vel_stats[0], softplus(vel_stats[1]), name='velocity_0')

        pos_stats = self.position_gibbs(stats)
        pos_stats = pos_stats.view(-1, direction.shape[1], 2, 2).unbind(dim=2)
        q.normal(pos_stats[0], softplus(pos_stats[1]), name='position_0')

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

        self.direction_gibbs = nn.Sequential(
            nn.Linear(2 * 7, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 4)
        )

        self.direction_feedback = nn.Sequential(
            nn.Linear(2 * 6, 16), nn.PReLU(),
            nn.Linear(16, 16), nn.PReLU(),
            nn.Linear(16, 2 * 4)
        )

    def forward(self, q, prev_dir, prev_pos, uncertainty, noise, next_dir,
                next_pos, data={}):
        velocity_stats = self.direction_gibbs(torch.cat(
            (prev_dir, prev_pos, data, next_dir, next_pos, uncertainty, noise),
            dim=2,
        )).view(-1, noise.shape[1], 2, 2)
        velocity_loc = velocity_stats[:, :, 0]
        velocity_scale = softplus(velocity_stats[:, :, 1])
        q.normal(loc=velocity_loc, scale=velocity_scale, name='velocity')

    def feedback(self, p, prev_dir, prev_pos, uncertainty, noise, data={}):
        stats = self.direction_feedback(torch.cat(
            (prev_dir, prev_pos, p['velocity'].value, p['position'].value,
             uncertainty, noise),
            dim=2,
        )).view(-1, noise.shape[1], 2, 4)
        return torch.unbind(stats, dim=-1)
