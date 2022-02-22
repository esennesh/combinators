#!/usr/bin/env python3

import logging
import torch

from combinators import sampler, signal
import combinators.inference.resample as resample
import combinators.inference.variational as variational

class ApgLoss(variational.VariationalLoss):
    def __init__(self, diagram):
        super().__init__(diagram)
        self._elbos = []
        self._eubos = []

    @property
    def elbos(self):
        return self._elbos

    @property
    def eubos(self):
        return self._eubos

    def objective(self):
        diagram = self._globals[0]
        _, log_weight = sampler.trace(diagram)

        theta_elbo = -variational.elbo(log_weight, iwae=True)
        self._elbos.append(theta_elbo)

        phi_eubo   = variational.eubo(log_weight, iwae=False)
        self._eubos.append(phi_eubo)

        return theta_elbo + phi_eubo

def hooks_apg(graph):
    resample.hook_resampling(graph, method='put', when='pre',
                             resampler_cls=resample.SystematicResampler)
    losses = ApgLoss(graph)
    variational.hook_variational(graph, losses, method='put', when='post')
    return losses

def apg(diagram, num_iterations, use_cuda=True, lr=1e-3, patience=50,
        num_sweeps=6):
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
    losses = hooks_apg(graph)

    filtering = sampler.filtering(graph)
    smoothing = sampler.smoothing(graph)
    theta, phi = sampler.parameters(graph)
    optimizer = torch.optim.Adam(theta | phi, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, min_lr=1e-6, patience=patience, mode='min'
    )

    objectives = torch.zeros(num_iterations, 2, requires_grad=False)
    for t in range(num_iterations):
        optimizer.zero_grad()

        filtering()
        for _ in range(num_sweeps):
            smoothing()

        loss = losses.loss
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        objectives[t, 0] = torch.stack(losses.elbos, dim=-1).detach()
        logging.info('Wake Theta IWAE free-energy=%.8e at epoch %d', loss,
                     t + 1)
        objectives[t, 1] = torch.stack(losses.eubos, dim=-1).detach()
        logging.info('Wake Phi EUBO=%.8e at epoch %d', loss, t + 1)

        sampler.clear(graph)

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
