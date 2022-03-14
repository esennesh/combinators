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
        self._particle_shape = particle_shape

    def objective(self):
        diagram = self._globals[0]
        _, log_weight = sampler.trace(diagram)

        theta_elbo = variational.elbo(log_weight, iwae=True,
                                      particle_shape=self._particle_shape)

        phi_eubo   = variational.eubo(log_weight, iwae=False,
                                      particle_shape=self._particle_shape)

        return -theta_elbo + phi_eubo

    def clear(self):
        super().clear()

def hooks_apg(graph, particle_shape):
    resample.hook_resampling(graph, particle_shape, method='put', when='pre',
                             resampler_cls=resample.SystematicResampler)
    losses = ApgLoss(graph, particle_shape)
    variational.hook_variational(graph, losses, method='put', when='post')
    return losses

def apg(diagram, num_iterations, particle_shape, data_loaders, use_cuda=True,
        lr=1e-3, patience=50, num_sweeps=6):
    data_loaders = list(data_loaders)
    graph = sampler.compile(diagram >> signal.Cap(diagram.cod))

    for box in graph:
        if isinstance(box, sampler.ImportanceWiringBox):
            box.sampler.train()

            if torch.cuda.is_available() and use_cuda:
                box.sampler.cuda()

    dims = tuple(range(len(particle_shape)))
    losses = hooks_apg(graph, particle_shape)

    filtering = sampler.filtering(graph)
    smoothing = sampler.smoothing(graph)
    theta, phi = sampler.parameters(graph)
    optimizer = torch.optim.Adam(theta | phi, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, min_lr=1e-6, patience=patience, mode='min'
    )

    log_joints = torch.zeros(num_iterations, requires_grad=False)
    for t in tqdm.tqdm(range(num_iterations)):
        for _, loader in tqdm.tqdm(data_loaders):
            for series in loader:
                if torch.cuda.is_available() and use_cuda:
                    series = series.cuda()

                conditioning.sequential(graph, step_where=series.unbind(dim=1))

                optimizer.zero_grad()

                filtering()
                for _ in range(num_sweeps):
                    smoothing()

                loss = losses.loss
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                p, _ = sampler.trace(graph)
                log_joint = p.log_joint(sample_dims=dims, batch_dim=len(dims))
                log_joints[t] += log_joint.detach().cpu().sum(dim=1).mean(dim=0)

                losses.clear()
                sampler.clear(graph)

                if torch.cuda.is_available() and use_cuda:
                    del series
                    torch.cuda.empty_cache()

        logging.info('Total log-joint=%.8e at epoch %d', log_joints[t].item(),
                     t + 1)

        if t % patience == 0:
            for box in graph:
                if isinstance(box, sampler.ImportanceWiringBox):
                    theta = box.sampler.target.state_dict()
                    torch.save(theta, box.name + '_theta_%d.pt' % (t+1))
                    phi = box.sampler.proposal.state_dict()
                    torch.save(phi, box.name + '_phi_%d.pt' % (t+1))

    for box in graph:
        if isinstance(box, sampler.ImportanceWiringBox):
            box.sampler.eval()
            if torch.cuda.is_available() and use_cuda:
                box.sampler.cpu()

    return log_joints
