#!/usr/bin/env python3

import collections
import itertools
import logging

import probtorch
from probtorch.util import log_sum_exp
import torch

import combinators
from combinators import GraphingTrace
import utils

EMPTY_ANNOTATION = collections.defaultdict(lambda: 0.0)

class ForwardMessenger(combinators.Model):
    def __init__(self, f, latent, observation, transition, observation_dists,
                 initial_marginals=None, phi={}, theta={}):
        self._latent = latent
        self._observation = observation
        self.transition = transition
        self.observation_dists = observation_dists
        self._initial_marginals = initial_marginals
        super(ForwardMessenger, self).__init__(f, phi=phi, theta=theta)

    def _condition(self, trace=None, observations=None):
        super(ForwardMessenger, self)._condition(trace, observations)
        if self._initial_marginals and (self._latent % 0 in self.trace):
            initial_note = self.trace.annotation(self._initial_marginals[0],
                                                 self._latent % 0)
            initial_note['forward_joint_marginals'] = self._initial_marginals[1]

    def forward(self, *args, **kwargs):
        results = super(ForwardMessenger, self).forward(*args, **kwargs)

        self.forward_pass(args[-1] + 1)

        return results

    def forward_pass(self, t):
        num_particles = self.trace[self._latent % t].value.shape[0]
        if t == 1:
            prev_note = self.trace.annotation(self._initial_marginals[0],
                                              self._latent % (t-1))
        else:
            prev_note = self.trace.annotation(self.name,
                                              self._latent % (t-1))
        latent_note = self.trace.annotation(self.name, self._latent % t)
        support = self.trace[self._latent % t].dist.enumerate_support()
        support = range(len(support))

        marginals = torch.zeros(num_particles, len(support), len(support))
        for prev, current in itertools.product(support, support):
            marginals[:, current, prev] =\
                self.transition(prev, current) +\
                prev_note['forward_joint_marginals'][:, prev]
        marginals = log_sum_exp(marginals, dim=-1)

        alpha = torch.zeros(num_particles, len(support))
        observed = self.trace[self._observation % t].value
        for current in support:
            alpha[:, current] =\
                self.observation_dists[current].log_prob(observed) +\
                marginals[:, current]
        latent_note['forward_joint_marginals'] = alpha

class ForwardBackwardMessenger(ForwardMessenger):
    def _backward_step(self, t, T):
        support = self.trace[self._latent % t].dist.enumerate_support()
        support = range(len(support))
        num_particles = self.trace[self._latent % t].value.shape[0]

        if t == T:
            final_note = self.trace.annotation(self.name, self._latent % t)
            final_note['backward_joint_marginals'] = torch.zeros(num_particles,
                                                                 len(support))
        else:
            next_note = self.trace.annotation(self.name, self._latent % (t+1))
            current_note = self.trace.annotation(self.name, self._latent % t)
            observed = self.trace[self._observation % t].value
            beta = torch.zeros(num_particles, len(support), len(support))
            for current, succ in itertools.product(support, support):
                beta[:, current, succ] =\
                    self.observation_dists[current].log_prob(observed) +\
                    self.transition(current, succ) +\
                    next_note['backward_joint_marginals'][:, succ]
            current_note['backward_joint_marginals'] = log_sum_exp(beta, dim=-1)

    def backward_pass(self, T):
        for t in reversed(range(T)):
            self._backward_step(t+1, T)

    def _posterior_step(self, t):
        note = self.trace.annotation(self.name, self._latent % (t+1))
        return note['forward_joint_marginals']+note['backward_joint_marginals']

    def smoothed_posterior(self, T):
        posteriors = torch.stack([self._posterior_step(n) for n in range(T)],
                                 dim=0)
        posteriors = posteriors.transpose(0, 1)
        denominators = log_sum_exp(posteriors, dim=-1)
        return posteriors - denominators.unsqueeze(-1), denominators

def variational_forward_backward(model_init, step_builder, num_iterations, T,
                                 data, *args, use_cuda=True, lr=1e-6):
    optimizer = torch.optim.Adam(list(model_init.parameters()), lr=lr)

    if torch.cuda.is_available() and use_cuda:
        model_init.cuda()

    for t in range(num_iterations):
        optimizer.zero_grad()

        inference = GraphingTrace()
        model_init.condition(trace=inference, observations=data)

        vs = model_init(*args, T)
        model_step = step_builder(*vs)
        model_step.condition(trace=inference, observations=data)
        if torch.cuda.is_available() and use_cuda:
            model_step.cuda()

        sequencer = combinators.Model.sequence(model_step, T, *vs)
        sequencer.condition(trace=inference, observations=data)
        vs = sequencer()

        model_step.backward_pass(T)
        _, marginals = model_step.smoothed_posterior(T)
        elbo = marginals.sum(dim=-1)
        logging.info('Variational forward-backward ELBO=%.8e at epoch %d',
                     elbo, t + 1)
        (-elbo.sum()).backward()
        optimizer.step()

    if torch.cuda.is_available() and use_cuda:
        model_init.cpu()
        model_step.cpu()

    return inference, model_init.args_vardict()
