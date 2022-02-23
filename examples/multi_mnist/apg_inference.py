#!/usr/bin/env python3

import logging
import torch
import torch.nn as nn
import tqdm

from combinators import sampler, signal
import combinators.inference.conditioning as conditioning
import combinators.inference.resample as resample
import combinators.inference.variational as variational

class ApgLoss(variational.VariationalLoss):
    def __init__(self, diagram, particle_shape=(1,)):
        super().__init__(diagram)
        self._elbos = []
        self._eubos = []
        self._particle_shape = particle_shape

    @property
    def elbos(self):
        return self._elbos

    @property
    def eubos(self):
        return self._eubos

    def objective(self):
        diagram = self._globals[0]
        _, log_weight = sampler.trace(diagram)

        theta_elbo = -variational.elbo(log_weight, iwae=True,
                                       particle_shape=self._particle_shape)
        self._elbos.append(theta_elbo)

        phi_eubo   = variational.eubo(log_weight, iwae=False,
                                      particle_shape=self._particle_shape)
        self._eubos.append(phi_eubo)

        return theta_elbo + phi_eubo

def hooks_apg(graph, particle_shape):
    resample.hook_resampling(graph, particle_shape, method='put', when='pre',
                             resampler_cls=resample.SystematicResampler)
    losses = ApgLoss(graph, particle_shape)
    variational.hook_variational(graph, losses, method='put', when='post')
    return losses

def apg(diagram, num_iterations, particle_shape, data_loaders, use_cuda=True,
        lr=1e-3, patience=50, num_sweeps=6):
    for box in diagram:
        if isinstance(box, sampler.ImportanceWiringBox):
            box.target.train()
            box.proposal.train()

    if torch.cuda.is_available() and use_cuda:
        for box in diagram:
            if isinstance(box, sampler.ImportanceWiringBox):
                box.target.cuda()
                box.proposal.cuda()

    graph = sampler.compile(diagram >> signal.Cap(diagram.cod))
    losses = hooks_apg(graph, particle_shape)

    filtering = sampler.filtering(graph)
    smoothing = sampler.smoothing(graph)
    theta, phi = sampler.parameters(graph)
    optimizer = torch.optim.Adam(theta | phi, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, min_lr=1e-6, patience=patience, mode='min'
    )

    objectives = torch.zeros(num_iterations, 2, requires_grad=False)
    for t in range(num_iterations):
        for _, loader in data_loaders:
            for series in loader:
                conditioning.sequential(graph, step_where=series.unbind(dim=1))

                optimizer.zero_grad()

                filtering()
                for _ in range(num_sweeps):
                    smoothing()

                loss = losses.loss
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                objectives[t, 0] += torch.stack(losses.elbos, dim=-1).detach()
                objectives[t, 1] += torch.stack(losses.eubos, dim=-1).detach()

                sampler.clear(graph)

        logging.info('Wake Theta IWAE free-energy=%.8e at epoch %d', loss,
                     t + 1)
        logging.info('Wake Phi EUBO=%.8e at epoch %d', loss, t + 1)

        if t % patience == 0:
            for box in diagram:
                if isinstance(box, sampler.ImportanceWiringBox):
                    theta = nn.utils.parameters_to_vector(box.target)
                    torch.save(theta, box.name + '_theta_%d.pt' % t)
                    phi = nn.utils.parameters_to_vector(box.proposal)
                    torch.save(phi, box.name + '_phi_%d.pt' % t)

    if torch.cuda.is_available() and use_cuda:
        for box in diagram:
            if isinstance(box, sampler.ImportanceWiringBox):
                box.proposal.cpu()
                box.target.cpu()

    for box in diagram:
        if isinstance(box, sampler.ImportanceWiringBox):
            box.proposal.eval()
            box.target.eval()

    return objectives
