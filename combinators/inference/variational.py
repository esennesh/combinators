#!/usr/bin/env python3

from abc import abstractmethod
import logging
import torch
import torch.nn.functional as F

from .. import lens, sampler, signal, utils

def elbo(log_weight, particle_shape=(1,), iwae=False):
    if iwae:
        l = utils.batch_marginalize(log_weight, particle_shape)
    else:
        l = utils.batch_mean(log_weight, particle_shape)
    return l.mean()

def eubo(log_weight, iwae=False):
    probs = utils.normalize_weights(log_weight).detach()
    particles = probs * log_weight
    if iwae:
        return utils.log_sum_exp(particles)
    return utils.batch_sum(eubo)

def infer(diagram, num_iterations, objective=elbo, use_cuda=True, lr=1e-3,
          patience=50):
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
    filtering = sampler.filtering(graph)
    smoothing = sampler.smoothing(graph)
    theta, phi = sampler.parameters(graph)
    optimizer = torch.optim.Adam(theta | phi, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, min_lr=1e-6, patience=patience, mode='max'
    )

    objs = torch.zeros(num_iterations, requires_grad=False)
    for t in range(num_iterations):
        optimizer.zero_grad()

        filtering()
        smoothing()

        _, log_weight = sampler.trace(graph)
        loss = objective(log_weight)

        (-loss).backward()
        optimizer.step()

        loss = loss.item()
        logging.info('%s=%.8e at epoch %d', objective.__name__, loss, t + 1)
        scheduler.step(loss)
        objs[t] = loss

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

    return objs

class VariationalLoss:
    def __init__(self, *args):
        self._globals = args
        self._loss = 0.

    @property
    def loss(self):
        return self._loss

    @abstractmethod
    def objective(self, *args):
        pass

    def accumulate(self, *vals):
        self._loss = self._loss + self.objective(*vals)
        return vals

def hook_variational(graph, variational, method='put', when='post'):
    for box in graph:
        if isinstance(box, sampler.ImportanceWiringBox):
            kwargs = {when + '_' + method: variational.accumulate}
            lens.hook(box, **kwargs)
