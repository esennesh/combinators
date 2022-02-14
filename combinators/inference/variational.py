#!/usr/bin/env python3

import logging
import torch
import torch.nn.functional as F

from .. import sampler, signal, utils

def elbo(log_weight, iwae=False):
    if iwae:
        return utils.batch_marginalize(log_weight)
    return utils.batch_mean(log_weight)

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

    graph = sampler.compile(diagram)
    theta, phi = sampler.parameters(graph)
    optimizer = torch.optim.Adam(theta | phi, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, min_lr=1e-6, patience=patience, mode='max'
    )

    objs = torch.zeros(num_iterations, requires_grad=False)
    for t in range(num_iterations):
        optimizer.zero_grad()
        sampler.filter(graph)

        feedback = signal.Signal.id(len(graph.cod))
        sampler.smooth(graph, *feedback.split())
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
