#!/usr/bin/env python3

import gym
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Normal, OneHotCategorical
from torch.distributions import RelaxedOneHotCategorical
from torch.distributions.transforms import LowerCholeskyTransform
import torch.nn as nn
from torch.nn.functional import softplus

import combinators.model as model

class CartpoleStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 4)
        self._action_dim = kwargs.pop('action_dim', 2)
        self._observation_dim = kwargs.pop('observation_dim', 4)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'next_state': {
                    'loc': torch.zeros(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
                'control': {
                    'probs': torch.ones(self._action_dim),
                },
                'observation_noise': {
                    'loc': torch.ones(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
            }
            kwargs['params']['next_state']['scale'][0] = 2.4 / 2
            kwargs['params']['next_state']['scale'][2] = np.pi / (15 * 2)
        super(CartpoleStep, self).__init__(*args, **kwargs)

    def _forward(self, theta, t, env=None):
        prev_state = theta

        control = self.param_sample(OneHotCategorical, name='control')
        state = self.param_sample(Normal, name='next_state')

        observation, _, _, _ = env.retrieve_step(
            t + 1, control.argmax(dim=-1)[0].item(), override_done=True
        )
        if observation is not None:
            observation = torch.Tensor(observation).to(state.device).expand(
                self.batch_shape + observation.shape
            )
        observation_noise = self.param_sample(Normal, 'observation_noise')
        state = self.observe('observation_%d' % (t+1), observation, Normal,
                             state, softplus(observation_noise))

        return state

class MountainCarStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'next_state': {
                    'loc': torch.zeros(self._observation_dim),
                    'scale': torch.ones(self._observation_dim),
                },
                'control': {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                },
                'observation_noise': {
                    'loc': torch.ones(self._observation_dim) * 1e-4,
                    'scale': torch.ones(self._observation_dim),
                },
            }
            kwargs['params']['next_state']['loc'][0] = 0.45
            kwargs['params']['next_state']['scale'][0] = 0.045
            kwargs['params']['next_state']['scale'][1] = 0.14 / 3
        super(MountainCarStep, self).__init__(*args, **kwargs)

    def _forward(self, theta, t, env=None):
        control = self.param_sample(Normal, name='control').expand(
            *self.batch_shape, 1
        )
        state = self.param_sample(Normal, 'next_state')

        if isinstance(control, torch.Tensor):
            action = control[0].cpu().detach().numpy()
        else:
            action = control
        observation, _, done, _ = env.retrieve_step(t, action,
                                                    override_done=True)
        if observation is not None and not done:
            observation = torch.Tensor(observation).to(state.device).expand(
                self.batch_shape + observation.shape
            )
        else:
            observation = None
        observation_noise = self.param_sample(Normal, 'observation_noise')
        state = self.observe('observation', observation, Normal, state,
                             softplus(observation_noise))

        return state

class GenerativeStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'state_uncertainty': {
                    'loc': torch.zeros(self._state_dim),
                    'covariance_matrix': torch.eye(self._state_dim),
                },
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
                'control_0': {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                },
            }
            if self._discrete_actions:
                kwargs['params']['control'] = {
                    'probs': torch.ones(self._action_dim)
                }
            else:
                kwargs['params']['control'] = {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
        super(GenerativeStep, self).__init__(*args, **kwargs)
        self.register_buffer('goal__loc', torch.zeros(self._observation_dim,
                                                      requires_grad=True))
        self.register_buffer('goal__scale', torch.ones(self._observation_dim,
                                                       requires_grad=True))
        self.state_transition = nn.Sequential(
            nn.Linear(self._state_dim + self._action_dim * 2,
                      self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim),
        )
        self.predict_observation = nn.Sequential(
            nn.Linear(self._state_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._observation_dim),
        )

    def _forward(self, theta, t, env=None):
        if theta is None:
            prev_state = self.param_sample(Normal, 'state_0')
            prev_control = self.param_sample(Normal, 'control_0')
        else:
            prev_state, prev_control = theta
        state_uncertainty = self.param_sample(MultivariateNormal,
                                              name='state_uncertainty')

        if self._discrete_actions:
            control = self.param_sample(OneHotCategorical, name='control')
        else:
            control = prev_control + self.param_sample(Normal, name='control')
            control = control.expand(*self.batch_shape, self._action_dim)

        state = self.state_transition(torch.cat(
            (prev_state, prev_control, control), dim=-1
        ))
        state = state + state_uncertainty

        if isinstance(control, torch.Tensor):
            action = control[0].cpu().detach().numpy()
        else:
            action = control
        observation, _, done, _ = env.retrieve_step(t, action,
                                                    override_done=True)
        if done:
            _, _, prev_done, _ = env.retrieve_step(t-1, None)
            finished = done and not prev_done
        else:
            finished = False
        if observation is not None and not done:
            observation = torch.Tensor(observation).to(state.device).expand(
                self.batch_shape + observation.shape
            )
        else:
            observation = None

        if not done or finished:
            prediction = self.predict_observation(state)
            observation_noise = self.param_sample(Normal,
                                                  name='observation_noise')
            observation_scale = LowerCholeskyTransform()(observation_noise)
            self.observe('observation', observation, MultivariateNormal,
                         prediction, scale_tril=observation_scale)
            self.observe('goal', prediction, Normal, self.goal__loc,
                         softplus(self.goal__scale))

        return state, control

class RecognitionStep(model.Primitive):
    def __init__(self, *args, **kwargs):
        self._state_dim = kwargs.pop('state_dim', 2)
        self._action_dim = kwargs.pop('action_dim', 1)
        self._observation_dim = kwargs.pop('observation_dim', 2)
        self._discrete_actions = kwargs.pop('discrete_actions', True)
        self._name = kwargs.pop('name')
        if 'params' not in kwargs:
            kwargs['params'] = {
                'state_0': {
                    'loc': torch.zeros(self._state_dim),
                    'scale': torch.ones(self._state_dim),
                },
                'observation_noise': {
                    'loc': torch.eye(self._observation_dim),
                    'scale': torch.ones(self._observation_dim,
                                        self._observation_dim),
                },
                'control_0': {
                    'loc': torch.zeros(self._action_dim),
                    'scale': torch.ones(self._action_dim),
                }
            }
        super(RecognitionStep, self).__init__(*args, **kwargs)
        if self._discrete_actions:
            self.decode_policy = nn.Sequential(
                nn.Linear(self._state_dim + self._action_dim,
                          self._state_dim * 4),
                nn.PReLU(),
                nn.Linear(self._state_dim * 4, self._action_dim),
                nn.Softmax(dim=-1),
            )
        else:
            self.decode_policy = nn.Sequential(
                nn.Linear(self._state_dim + self._action_dim, 8),
                nn.PReLU(),
                nn.Linear(self._state_dim * 4, self._state_dim * 8),
                nn.PReLU(),
                nn.Linear(self._state_dim * 8, self._state_dim * 16),
                nn.PReLU(),
                nn.Linear(self._state_dim * 16, self._action_dim * 2),
                nn.Softsign(),
            )
        self.encode_uncertainty = nn.Sequential(
            nn.Linear(self._state_dim + self._observation_dim +
                      self._action_dim, self._state_dim * 4),
            nn.PReLU(),
            nn.Linear(self._state_dim * 4, self._state_dim * 8),
            nn.PReLU(),
            nn.Linear(self._state_dim * 8, self._state_dim * 16),
            nn.PReLU(),
            nn.Linear(self._state_dim * 16, self._state_dim * 2),
        )

    @property
    def name(self):
        return self._name

    def _forward(self, theta, t, env=None):
        if theta is None:
            prev_state = self.param_sample(Normal, 'state_0')
            prev_control = self.param_sample(Normal, 'control_0')
        else:
            prev_state, prev_control = theta
        self.param_sample(Normal, name='observation_noise')

        control = self.decode_policy(
            torch.cat((prev_state, prev_control), dim=-1)
        )
        if self._discrete_actions:
            control = self.sample(OneHotCategorical, probs=control,
                                  name='control')
        else:
            control = control.reshape(-1, self._action_dim, 2)
            control = prev_control + self.sample(Normal, control[:, :, 0],
                                                 softplus(control[:, :, 1]),
                                                 name='control')

        if isinstance(control, torch.Tensor):
            action = control[0].cpu().detach().numpy()
        else:
            action = control
        observation, _, done, _ = env.retrieve_step(t, action,
                                                    override_done=True)
        if observation is not None and not done:
            observation = torch.Tensor(observation).to(control.device).expand(
                self.batch_shape + observation.shape
            )
        else:
            observation = torch.zeros(
                self.batch_shape + (self._observation_dim,)
            ).to(control.device)
        state_uncertainty = self.encode_uncertainty(
            torch.cat((prev_state, observation, control), dim=-1)
        ).reshape(-1, self._state_dim, 2)
        self.sample(Normal, state_uncertainty[:, :, 0],
                    softplus(state_uncertainty[:, :, 1]),
                    name='state_uncertainty')
