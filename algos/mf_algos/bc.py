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

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad


class BC(Algo):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.target_entropy = 0.0
    config.policy_lr = 1e-4
    config.optimizer_type = 'adam'
    config.bc_mode = 'mse'  # 'mle', 'mse'
    config.use_scheduler = False
    config.max_target_backup = False

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf, max_steps):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.observation_dim = policy.input_size
    self.action_dim = policy.action_dim
    self.max_steps = max_steps

    self._train_states = {}

    optimizer_class = {
      'adam': optax.adam,
      'sgd': optax.sgd,
    }[self.config.optimizer_type]

    policy_params = self.policy.init(
      {
        'dropout': next_rng(),
        'params': next_rng()
      },
      next_rng(), jnp.zeros((10, self.observation_dim))
    )
    policy_optim = optimizer_class(self.config.policy_lr)
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

    model_keys = ['policy']

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch, update_target_policy):
    self._total_steps += 1
    self._train_states, metrics = self._train_step(
      self._train_states, next_rng(), batch
    )
    return metrics

  @partial(jax.jit, static_argnames=('self'))
  def _train_step(self, train_states, rng, batch):

    def loss_fn(train_params, rng):
      observations = batch['observations']
      actions = batch['actions']

      loss_collection = {}

      rng, split_rng = jax.random.split(rng)

      new_actions, _ = self.policy.apply(
        train_params['policy'], split_rng, observations,
        rngs={'dropout': split_rng}
      )
      """ Policy loss """
      # get bc loss
      if self.config.bc_mode == 'mle':
        rng, split_rng = jax.random.split(rng)
        log_probs = self.policy.apply(
          train_params['policy'],
          observations,
          actions,
          method=self.policy.log_prob,
          rngs={'dropout': split_rng}
        )
        bc_loss = -log_probs.mean()
      elif self.config.bc_mode == 'mse':
        bc_loss = mse_loss(actions, new_actions)
      else:
        raise RuntimeError('{} not implemented!'.format(self.config.bc_mode))

      # total loss for policy
      policy_loss = bc_loss
      loss_collection['policy'] = policy_loss
      loss_collection['bc_loss'] = bc_loss

      return tuple(loss_collection[key] for key in self.model_keys), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(train_params, rng)

    new_train_states = {
      key: train_states[key].apply_gradients(grads=grads[i][key])
      for i, key in enumerate(self.model_keys)
    }

    metrics = dict(
      policy_loss=aux_values['policy_loss'],
      bc_loss=aux_values['bc_loss'],
    )

    return new_train_states, metrics

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
