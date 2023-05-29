# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.mf_algos.model import update_target_network
from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad


class TD3(Algo):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.tau = 0.005
    config.policy_noise = 0.2
    config.noise_clip = 0.5
    config.policy_freq = 2
    config.alpha = 2.5
    config.policy_lr = 3e-4
    config.qf_lr = 3e-4
    config.use_scheduler = False
    config.target_entropy = 0.0

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf, max_steps):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim
    self.policy_noise = self.config.policy_noise * policy.max_action
    self.max_action = policy.max_action
    self.max_step = max_steps

    self._total_steps = 0
    self._train_states = {}

    policy_params = self.policy.init(
      {
        'dropout': next_rng(),
        'params': next_rng()
      },
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
    )
    policy_optim = optax.adam(self.config.policy_lr)
    if self.config.use_scheduler:
      schedule_fn = optax.cosine_decay_schedule(
        -self.config.policy_lr, self.max_steps
      )
      policy_optim = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
      )
    self._train_states['policy'] = TrainState.create(
      params=policy_params, tx=policy_optim, apply_fn=None
    )

    qf1_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    self._train_states['qf1'] = TrainState.create(
      params=qf1_params,
      tx=optax.adam(self.config.qf_lr),
      apply_fn=None,
    )
    qf2_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    self._train_states['qf2'] = TrainState.create(
      params=qf2_params,
      tx=optax.adam(self.config.qf_lr),
      apply_fn=None,
    )

    self._tgt_params = deepcopy(
      {
        'policy': policy_params,
        'qf1': qf1_params,
        'qf2': qf2_params,
      }
    )

    self._model_keys = ('policy', 'qf1', 'qf2')

  @partial(jax.jit, static_argnames=('self', 'policy_update'))
  def _train_step(
    self, train_states, tgt_params, rng, batch, policy_update=False
  ):

    def loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      rng, split_rng = jax.random.split(rng)
      # Select next actions according to policy and add noise
      noise = jax.random.truncated_normal(
        split_rng,
        -self.config.noise_clip,
        self.config.noise_clip,
        shape=actions.shape,
        dtype=actions.dtype,
      ) * self.policy_noise

      rng, split_rng = jax.random.split(rng)
      next_action = self.policy.apply(
        params['policy'], rng, next_observations, rngs={'dropout': split_rng}
      )
      next_action = jnp.clip(
        next_action + noise, -self.max_action, self.max_action
      )

      # Compute the target Q values (without gradient)
      tgt_q1 = self.qf.apply(tgt_params['qf1'], next_observations, next_action)
      tgt_q2 = self.qf.apply(tgt_params['qf2'], next_observations, next_action)
      tgt_q = jnp.minimum(tgt_q1, tgt_q2)
      tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
      tgt_q = jax.lax.stop_gradient(tgt_q)

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # qf loss
      qf1_loss = mse_loss(cur_q1, tgt_q)
      qf2_loss = mse_loss(cur_q2, tgt_q)
      qf_loss = qf1_loss + qf2_loss

      # Compute the policy loss
      rng, split_rng = jax.random.split(rng)
      new_actions = self.policy.apply(
        params['policy'], rng, observations, rngs={'dropout': split_rng}
      )
      q = self.qf.apply(params['qf1'], observations, new_actions)
      q_abs_mean = jax.lax.stop_gradient(jnp.abs(q).mean())
      lmbda = self.config.alpha / q_abs_mean
      bc_loss = mse_loss(new_actions, actions)
      policy_loss = -lmbda * q.mean() + bc_loss

      loss_collection = {'policy': policy_loss, 'qf1': qf_loss, 'qf2': qf_loss}
      return tuple(loss_collection[key] for key in self.model_keys), locals()

    # Calculate losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads[1]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads[2]['qf2']
    )

    # Update policy train states if necessary
    if policy_update:
      train_states['policy'] = train_states['policy'].apply_gradients(
        grads=grads[0]['policy']
      )

      # Update target parameters
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
      tgt_params['qf1'] = update_target_network(
        train_states['qf1'].params, tgt_params['qf1'], self.config.tau
      )
      tgt_params['qf2'] = update_target_network(
        train_states['qf2'].params, tgt_params['qf2'], self.config.tau
      )

    metrics = dict(
      policy_loss=aux_values['policy_loss'],
      qf_loss=aux_values['qf_loss'],
      qf1_loss=aux_values['qf1_loss'],
      qf2_loss=aux_values['qf2_loss'],
      bc_loss=aux_values['bc_loss'],
      cur_q1=aux_values['cur_q1'].mean(),
      cur_q2=aux_values['cur_q2'].mean(),
      tgt_q1=aux_values['tgt_q1'].mean(),
      tgt_q2=aux_values['tgt_q2'].mean(),
      tgt_q=aux_values['tgt_q'].mean(),
      lmbda=aux_values['lmbda'],
      q_abs_mean=aux_values['q_abs_mean'],
    )

    return train_states, tgt_params, metrics

  def train(self, batch, update_target_policy):
    self._total_steps += 1
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states, self._tgt_params, next_rng(), batch,
      self._total_steps % self.config.policy_freq == 0
    )
    return metrics

  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    return {key: self.train_states[key].params for key in self.model_keys}

  @property
  def total_steps(self):
    return self._total_steps
