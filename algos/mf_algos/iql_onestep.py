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

from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad

from algos.mf_algos.iql import IQL

class IQL_Onestep(IQL):

  def train(self, batch, update_target_policy):
    raise NotImplementedError("self.train is disabled in onestep rl algorithms.")

  def train_beta(self, batch):
    raise NotImplementedError("IQL don't need explicit behavior policy.")

  def train_qf(self, batch):
    self._total_steps += 1
    self._train_states, self._target_qf_params, metrics = self._train_qf_step(
      self._train_states, self._target_qf_params, next_rng(), batch
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
  def _train_qf_step(self, train_states, target_qf_params, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    next_observations = batch['next_observations']
    dones = batch['dones']

    def value_loss(train_params):
      q1 = self.qf.apply(target_qf_params['qf1'], observations, actions)
      q2 = self.qf.apply(target_qf_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      v_pred = self.vf.apply(train_params['vf'], observations)
      diff = q_pred - v_pred
      expectile_weight = jnp.where(
        diff > 0,
        self.config.expectile,
        1 - self.config.expectile,
      )

      if self.config.loss_type == 'expectile':
        expectile_loss = (expectile_weight * (diff**2)).mean()
      elif self.config.loss_type == 'quantile':
        expectile_loss = (expectile_weight * (jnp.abs(diff))).mean()
      else:
        raise NotImplementedError

      return (expectile_loss,), locals()

    def critic_loss(train_params):
      next_v = self.vf.apply(train_params['vf'], next_observations)

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1 - dones) * discount * next_v
      )

      q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
      q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      return (qf1_loss, qf2_loss), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_value), value_grads = value_and_multi_grad(
      value_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['vf'] = train_states['vf'].apply_gradients(
      grads=value_grads[0]['vf']
    )

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), qf_grads = value_and_multi_grad(
      critic_loss, 2, has_aux=True
    )(
      train_params
    )

    for i, k in enumerate(['qf1', 'qf2']):
      train_states[k] = train_states[k].apply_gradients(grads=qf_grads[i][k])

    metrics = dict(
      q1_pred=aux_qf['q1_pred'].mean(),
      qf1_loss=aux_qf['qf1_loss'],
      td_target=aux_qf['td_target'].mean(),
    )

    new_target_qf_params = {}
    new_target_qf_params['qf1'] = update_target_network(
      train_states['qf1'].params, target_qf_params['qf1'],
      self.config.soft_target_update_rate
    )
    new_target_qf_params['qf2'] = update_target_network(
      train_states['qf2'].params, target_qf_params['qf2'],
      self.config.soft_target_update_rate
    )

    metrics = dict(
      v_pred=aux_value['v_pred'].mean(),
      q1_pred=aux_value['q1'].mean(),
      q2_pred=aux_value['q2'].mean(),
      expectile_weight=aux_value['expectile_weight'].mean(),
      expectile_loss=aux_value['expectile_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      td_target=aux_qf['td_target'].mean(),
    )

    return train_states, new_target_qf_params, metrics

  
  @partial(jax.jit, static_argnames=('self'))
  def _train_pi_step(self, train_states, target_qf_params, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    
    def awr_loss(train_params):
      v_pred = self.vf.apply(train_params['vf'], observations)
      q1 = self.qf.apply(target_qf_params['qf1'], observations, actions)
      q2 = self.qf.apply(target_qf_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      exp_a = jnp.exp((q_pred - v_pred) * self.config.awr_temperature)
      exp_a = jnp.minimum(exp_a, 100.0)
      log_probs = self.policy.apply(
        train_params['policy'],
        observations,
        actions,
        method=self.policy.log_prob,
        rngs={'dropout': rng}
      )
      awr_loss = -(exp_a * log_probs).mean()

      return (awr_loss,), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), policy_grads = value_and_multi_grad(
      awr_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=policy_grads[0]['policy']
    )

    metrics = dict(
      awr_exp_a=aux_policy['exp_a'].mean(),
      awr_log_prob=aux_policy['log_probs'].mean(),
      awr_loss=aux_policy['awr_loss'],
    )
    return train_states, metrics