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


class Onestep(Algo):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.target_entropy = 0.0
    config.beta_lr = 1e-4
    config.qf_lr = 1e-4
    config.pi_lr = 1e-4
    config.policy_lr = config.pi_lr
    config.optimizer_type = 'adam'
    config.bc_mode = 'mle'  # 'mle'
    config.target_update_freq = 2 
    config.soft_target_update_rate = 5e-3
    config.awr_temperature = 1.0
    config.use_scheduler = True

    config.beta = 'bc'
    config.qf = 'sarsa'
    config.pi = 'exp_weight'
    config.baseline = 'value_sample' # adv = q - baseline
    config.n_samples = 10
    config.double_q = False

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

    self.optimizer_class = {
      'adam': optax.adam,
      'sgd': optax.sgd,
    }[self.config.optimizer_type]

    beta_params = self.policy.init(
      {
        'dropout': next_rng(),
        'params': next_rng(),
      }, next_rng(), jnp.zeros((10, self.observation_dim))
    )
    beta_optim = self.optimizer_class(self.config.beta_lr)
    self._train_states['beta'] = TrainState.create(
      params=beta_params, tx=beta_optim, apply_fn=None
    )

    if self.config.double_q:
      qf1_params = self.qf.init(
        next_rng(), jnp.zeros((10, self.observation_dim)),
        jnp.zeros((10, self.action_dim))
      )
      self._train_states['qf1'] = TrainState.create(
        params=qf1_params,
        tx=self.optimizer_class(self.config.qf_lr),
        apply_fn=None,
      )
      qf2_params = self.qf.init(
        next_rng(), jnp.zeros((10, self.observation_dim)),
        jnp.zeros((10, self.action_dim))
      )
      self._train_states['qf2'] = TrainState.create(
        params=qf2_params,
        tx=self.optimizer_class(self.config.qf_lr),
        apply_fn=None,
      )
      self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})
      model_keys = ['beta', 'qf1', 'qf2']
    else:
      qf1_params = self.qf.init(
        next_rng(), jnp.zeros((10, self.observation_dim)),
        jnp.zeros((10, self.action_dim))
      )
      self._train_states['qf1'] = TrainState.create(
        params=qf1_params,
        tx=self.optimizer_class(self.config.qf_lr),
        apply_fn=None,
      )
      self._target_qf_params = deepcopy({'qf1': qf1_params})
      model_keys = ['beta', 'qf1']

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def init_pi(self):
    pi_params = deepcopy(self._train_states['beta'].params)
    pi_optim = self.optimizer_class(self.config.pi_lr)
    if self.config.use_scheduler:
      schedule_fn = optax.cosine_decay_schedule(
        -self.config.pi_lr, self.max_steps
      )
      pi_optim = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
      )
    self._train_states['pi'] = TrainState.create(
      params=pi_params, tx=pi_optim, apply_fn=None
    )
    model_keys = list(self._model_keys) + ['pi']
    self._model_keys = tuple(model_keys)

  def train(self, batch, **kwargs):
    raise NotImplementedError("self.train is disabled in onestep rl algorithms.")
  
  def train_beta(self, batch):
    self._total_steps += 1
    self._train_states, metrics = self._train_beta_step(
      self._train_states, next_rng(), batch
    )
    return metrics

  def train_qf(self, batch):
    self._total_steps += 1
    self._train_states, metrics = self._train_qf_step(
      self._train_states, self._target_qf_params, next_rng(), batch
    )
    if self.total_steps % self.config.target_update_freq == 0:
      self._target_qf_params['qf1'] = update_target_network(
          self._train_states['qf1'].params, self._target_qf_params['qf1'],
          self.config.soft_target_update_rate
        )
      if self.config.double_q:
        self._target_qf_params['qf2'] = update_target_network(
          self._train_states['qf2'].params, self._target_qf_params['qf2'],
          self.config.soft_target_update_rate
        )
    return metrics

  def train_pi(self, batch):
    self._total_steps += 1
    self._train_states, metrics = self._train_pi_step(
      self._train_states, self._target_qf_params, next_rng(), batch
    )
    return metrics


  def _train_step(self, train_state, target_params, rng, batch, **kwargs):
    raise NotImplementedError("self._train_step is disabled in onestep rl algorithms")
  
  @partial(jax.jit, static_argnames=('self'))
  def _train_beta_step(self, train_states, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    def bc_loss_fn(train_params, rng):
      rng, split_rng = jax.random.split(rng)
      if self.config.bc_mode == 'mle':
        log_probs = self.policy.apply(
          train_params['beta'],
          observations,
          actions,
          method=self.policy.log_prob,
          rngs={'dropout': split_rng}
        )
        bc_loss = -log_probs.mean()
      elif self.config.bc_mode == 'mse':
        new_actions, _ = self.policy.apply(
        train_params['beta'], split_rng, observations, rngs={'dropout': split_rng}
        )
        bc_loss = mse_loss(actions, new_actions)
      else:
        raise RuntimeError('{} not implemented!'.format(self.config.bc_mode))
      return (bc_loss,), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_beta), grads = value_and_multi_grad(
      bc_loss_fn, 1, has_aux=True
    )(train_params, rng)

    train_states['beta'] = train_states['beta'].apply_gradients(
      grads=grads[0]['beta']
    )

    metrics = dict(
      bc_loss=aux_beta['bc_loss'],
    )
    if 'log_prob' in metrics.keys():
      metrics['log_prob']=aux_beta['log_probs'].mean(),

    return train_states, metrics

  
  @partial(jax.jit, static_argnames=('self'))
  def _train_qf_step(self, train_states, target_qf_params, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    next_observations = batch['next_observations']
    next_actions = batch['next_actions']
    dones = batch['dones']

    def qf_loss_fn(train_params):
      if self.config.double_q:
        q1_next = self.qf.apply(train_params['qf1'], next_observations, next_actions)
        q2_next = self.qf.apply(train_params['qf2'], next_observations, next_actions)
        q_next = jnp.minimum(q1_next, q2_next)
      else:
        q_next = self.qf.apply(train_params['qf1'], next_observations, next_actions)

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1 - dones) * discount * q_next
      )

      if self.config.double_q:
        q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
        q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
        qf1_loss = mse_loss(q1_pred, td_target)
        qf2_loss = mse_loss(q2_pred, td_target)
      else:
        q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
        qf1_loss = mse_loss(q1_pred, td_target)
        qf2_loss = 0

      return (qf1_loss, qf2_loss), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    
    if self.config.double_q:
      (_, aux_qf), qf_grads = value_and_multi_grad(
        qf_loss_fn, 2, has_aux=True
      )(
        train_params
      )
      for i, k in enumerate(['qf1', 'qf2']):
        train_states[k] = train_states[k].apply_gradients(grads=qf_grads[i][k])
      
    else:
      (_, aux_qf), qf_grads = value_and_multi_grad(
        qf_loss_fn, 1, has_aux=True
      )(
        train_params
      )
      train_states['qf1'] = train_states['qf1'].apply_gradients(grads=qf_grads[0]['qf1'])

    metrics = dict(
      q1_pred=aux_qf['q1_pred'].mean(),
      qf1_loss=aux_qf['qf1_loss'],
      td_target=aux_qf['td_target'].mean(),
    )
    if self.config.double_q:
      metrics['q2_pred'] = aux_qf['q2_pred'].mean()
      metrics['qf2_loss'] = aux_qf['qf2_loss']

    return train_states, metrics

  
  @partial(jax.jit, static_argnames=('self'))
  def _train_pi_step(self, train_states, target_qf_params, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    
    def pi_loss_fn(train_params, rng):
      rng, split_rng = jax.random.split(rng)
      if self.config.baseline == 'value_sample':
        action_dist = self.policy.apply(
          train_params['beta'], observations,
          method=self.policy.get_action_dist,rngs={'dropout': split_rng}
        )
        samples = action_dist.sample(seed=split_rng, sample_shape=(self.config.n_samples,))
        samples = samples.reshape(-1, *samples.shape[2:])
        repeat_states = jnp.expand_dims(batch['observations'], 0).repeat(self.config.n_samples, axis=0)
        repeat_states = repeat_states.reshape(-1, *repeat_states.shape[2:])
        if self.config.double_q:
          q1_samples = self.qf.apply(train_params['qf1'], repeat_states, samples)
          q2_samples = self.qf.apply(train_params['qf2'], repeat_states, samples)
          q_samples = jnp.minimum(q1_samples, q2_samples)
        else:
          q_samples = self.qf.apply(train_params['qf1'], repeat_states, samples)
        q_samples = q_samples.reshape(self.config.n_samples, -1)
        baseline = q_samples.mean(0)
      else:
        raise  NotImplementedError
      
      if self.config.double_q:
        q1 = self.qf.apply(target_qf_params['qf1'], observations, actions)
        q2 = self.qf.apply(target_qf_params['qf2'], observations, actions)
        q = jnp.minimum(q1, q2)
      else:
        q = self.qf.apply(target_qf_params['qf1'], observations, actions)

      if self.config.pi == 'exp_weight':
        adv = jax.lax.stop_gradient(q - baseline)
        exp_a = jnp.exp(adv * self.config.awr_temperature)
        exp_a = jnp.minimum(exp_a, 100.0)
        rng, split_rng = jax.random.split(rng)
        log_probs = self.policy.apply(
          train_params['pi'],
          observations,
          actions,
          method=self.policy.log_prob,
          rngs={'dropout': split_rng}
        )
        pi_loss = -(exp_a * log_probs).mean()
      else:
        raise NotImplementedError

      return (pi_loss,), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), policy_grads = value_and_multi_grad(
      pi_loss_fn, 1, has_aux=True
    )(
      train_params, rng
    )
    train_states['pi'] = train_states['pi'].apply_gradients(
      grads=policy_grads[0]['pi']
    )

    metrics = dict(
      awr_exp_a=aux_policy['exp_a'].mean(),
      awr_log_prob=aux_policy['log_probs'].mean(),
      awr_loss=aux_policy['pi_loss'],
    )

    return train_states, metrics


  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    param_dict = {key: self.train_states[key].params for key in self.model_keys}
    param_dict['policy'] = self.train_states['beta'].params
    return param_dict

  @property
  def total_steps(self):
    return self._total_steps
