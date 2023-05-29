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

"""
NN models.
"""

from functools import partial
from typing import Optional

import distrax
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from utilities.jax_utils import extend_and_repeat, next_rng

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


def update_target_network(main_params, target_params, tau):
  return jax.tree_map(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


def multiple_action_q_function(forward):

  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1])
      observations = observations.reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values = forward(self, observations, actions, **kwargs)
    if multiple_actions:
      q_values = q_values.reshape(batch_size, -1)
    return q_values

  return wrapped


class Scalar(nn.Module):
  init_value: float

  def setup(self):
    self.value = self.param("value", lambda x: self.init_value)

  def __call__(self):
    return self.value


class FullyConnectedNetwork(nn.Module):
  output_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = "relu"
  dropout: Optional[float] = None
  last_layer_init: Optional[float] = 0.01

  @nn.compact
  def __call__(self, input_tensor, training=True):
    x = input_tensor
    hidden_sizes = [int(h) for h in self.arch.split("-")]
    for h in hidden_sizes:
      if self.orthogonal_init:
        x = nn.Dense(
          h,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(h)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      if self.dropout is not None:
        x = nn.Dropout(self.dropout)(x, deterministic=not training)
      x = getattr(nn, self.activation)(x)

    if self.orthogonal_init:
      output = nn.Dense(
        self.output_dim,
        kernel_init=jax.nn.initializers.orthogonal(self.last_layer_init,),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      output = nn.Dense(
        self.output_dim,
        kernel_init=jax.nn.initializers.variance_scaling(
          self.last_layer_init, "fan_in", "uniform"
        ),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )

    if self.use_layer_norm:
      x = nn.LayerNorm()(x)

    return output


class FullyConnectedQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = "relu"
  last_layer_init: Optional[float] = 0.01

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)
    x = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation,
      last_layer_init=self.last_layer_init,
    )(
      x
    )
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim


class FullyConnectedVFunction(nn.Module):
  observation_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = 'relu'
  last_layer_init: Optional[float] = 0.01

  @nn.compact
  def __call__(self, observations):
    x = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation,
      last_layer_init=self.last_layer_init,
    )(
      observations
    )

    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim


class TanhGaussianPolicy(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  use_layer_norm: bool = False
  state_dependent_std: bool = True
  tanh_squash_distribution: bool = True
  clip_mean: bool = True
  log_sig_max: Optional[float] = None
  log_sig_min: Optional[float] = None
  use_log_std_multiplier: bool = True
  dropout: Optional[float] = None
  last_layer_init: Optional[float] = 0.01

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      dropout=self.dropout,
      last_layer_init=self.last_layer_init,
    )
    if self.state_dependent_std:
      self.std_network = FullyConnectedNetwork(
        output_dim=self.action_dim,
        arch=self.arch,
        orthogonal_init=self.orthogonal_init,
        use_layer_norm=self.use_layer_norm,
        dropout=self.dropout,
        last_layer_init=self.last_layer_init,
      )
    else:
      self.std = self.param(
        'log_std', nn.initializers.zeros, (self.action_dim,)
      )
      self.std_network = lambda x, training: self.std

    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)

  def get_dist_params(self, observations, repeat=None, training=True):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    mean = self.base_network(observations, training=training)
    log_std = self.std_network(observations, training=training)
    if self.clip_mean:
      mean = jnp.clip(mean, MEAN_MIN, MEAN_MAX)

    if self.use_log_std_multiplier:
      log_std = (
        self.log_std_multiplier_module() * log_std +
        self.log_std_offset_module()
      )

    if self.log_sig_max and self.log_sig_min:
      log_sig_min, log_sig_max = self.log_sig_min, self.log_sig_max
    else:
      log_sig_min, log_sig_max = LOG_SIG_MIN, LOG_SIG_MAX

    log_std = jnp.clip(log_std, log_sig_min, log_sig_max)

    if self.tanh_squash_distribution:
      action_distribution = distrax.Transformed(
        distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
        distrax.Block(distrax.Tanh(), ndims=1),
      )
    else:
      mean = nn.tanh(mean)
      action_distribution = distrax.MultivariateNormalDiag(
        mean, jnp.exp(log_std)
      )

    return action_distribution, mean, log_std

  def get_action_dist(self, observations, repeat=None, training=True):
    action_distribution, _, _ = self.get_dist_params(
      observations, repeat, training=training
    )
    return action_distribution

  def log_prob(self, observations, actions, training=True):
    if actions.ndim == 3:
      observations = extend_and_repeat(observations, 1, actions.shape[1])
    action_distribution = self.get_action_dist(observations, training=training)
    return action_distribution.log_prob(actions)

  def __call__(
    self, rng, observations, deterministic=False, repeat=None, training=True
  ):
    action_distribution, mean, _ = self.get_dist_params(
      observations, repeat, training=training
    )

    if deterministic:
      if self.tanh_squash_distribution:
        samples = jnp.tanh(mean)
      else:
        samples = mean
      log_prob = action_distribution.log_prob(samples)
    else:
      samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

    return samples, log_prob

  @property
  def input_size(self):
    return self.observation_dim


class SamplerPolicy(object):

  def __init__(self, policy, policy_params, mean=0, std=1):
    self.policy = policy
    self.policy_params = policy_params
    self.mean = mean
    self.std = std

  def update_params(self, params):
    self.params = params
    return self

  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
      params, rng, observations, deterministic, repeat=None, training=False
    )

  def __call__(self, observations, deterministic=False):
    observations = (observations - self.mean) / self.std
    actions = self.act(
      self.params, next_rng(), observations, deterministic=deterministic
    )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DirectMappingPolicy(nn.Module):
  observation_dim: int
  action_dim: int
  max_action: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  dropout: Optional[float] = None
  last_layer_init: Optional[float] = 0.01

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      dropout=self.dropout,
      last_layer_init=self.last_layer_init,
    )

  def __call__(
    self, rng, observations, deterministic=True, repeat=None, training=True
  ):
    # `rng` and `deterministic` are ununsed parameters
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    action = self.base_network(observations, training=training)
    return jnp.tanh(action) * self.max_action


class ClipGaussianPolicy(TanhGaussianPolicy):
  observation_dim: int
  action_dim: int
  arch: str = '512-512-256'
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  use_layer_norm: bool = True
  activation: str = 'elu'
  dropout: Optional[float] = None
  last_layer_init: Optional[float] = 0.01

  def __call__(
    self, rng, observations, deterministic=False, repeat=None, training=True
  ):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    base_network_output = self.base_network(observations, training=training)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    mean = nn.tanh(mean)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    action_distribution = distrax.MultivariateNormalDiag(
      mean, jnp.exp(log_std)
    )
    if deterministic:
      samples = mean
      log_prob = action_distribution.log_prob(samples)
    else:
      samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

    samples = jnp.clip(samples, -1, 1)
    return samples, log_prob

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=2 * self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation,
      last_layer_init=self.last_layer_init,
    )
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)

  def get_tfd_dist(self, observations):
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    mean = nn.tanh(mean)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
    return dist


class LinearResBlockV1(nn.Module):
  hidden_size: int
  use_projection: bool = True

  @nn.compact
  def __call__(self, x):
    shortcut = x
    if self.use_projection:
      shortcut = nn.Dense(self.hidden_size, use_bias=False)(shortcut)
      shortcut = nn.LayerNorm()(shortcut)

    x = nn.Dense(self.hidden_size, use_bias=False)(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size, use_bias=False)(x)
    x = nn.LayerNorm()(x)
    return nn.relu(x + shortcut)


class LinearResBlockV2(nn.Module):
  hidden_size: int
  use_projection: bool = True

  @nn.compact
  def __call__(self, x):
    shortcut = x

    x = nn.LayerNorm()(x)
    x = nn.relu(x)

    if self.use_projection:
      shortcut = nn.Dense(self.hidden_size, use_bias=False)(x)

    x = nn.Dense(self.hidden_size, use_bias=False)(x)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size, use_bias=False)(x)

    return x + shortcut


class LinearResBlockCRR(nn.Module):
  hidden_size: int
  use_projection: bool = True

  @nn.compact
  def __call__(self, x):
    shortcut = x

    if self.use_projection:
      shortcut = nn.Dense(self.hidden_size, use_bias=False)(x)

    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)
    x = nn.Dense(self.hidden_size)(x)
    x = nn.relu(x)

    return nn.LayerNorm()(x + shortcut)


class ResNetwork(nn.Module):
  output_dim: int
  hidden_dim: int
  use_projection: bool = True
  num_blocks: int = 1
  res_type: str = 'v2'

  @nn.compact
  def __call__(self, input_tensor):
    x = input_tensor

    for i in range(self.num_blocks):
      use_projection = self.use_projection if i == 0 else False
      if self.res_type == 'v1':
        x = LinearResBlockV1(self.hidden_dim, use_projection)(x)
      elif self.res_type == 'v2':
        x = LinearResBlockV2(self.hidden_dim, use_projection)(x)
      elif self.res_type == 'crr':
        x = LinearResBlockCRR(self.hidden_dim, use_projection)(x)

    if self.res_type in ['v2', 'crr']:
      x = nn.relu(x)

    x = nn.Dense(self.output_dim)(x)

    return x


class ResQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  hidden_dim: int = 256
  num_blocks: int = 1
  res_type: str = 'v2'

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)
    x = ResNetwork(
      output_dim=1,
      hidden_dim=self.hidden_dim,
      num_blocks=self.num_blocks,
      res_type=self.res_type,
    )(
      x
    )
    return jnp.squeeze(x, -1)


class ResVFunction(nn.Module):
  observation_dim: int
  hidden_dim: int = 256
  num_blocks: int = 1
  res_type: str = 'v2'

  @nn.compact
  def __call__(self, observations):
    x = ResNetwork(
      output_dim=1,
      hidden_dim=self.hidden_dim,
      num_blocks=self.num_blocks,
      res_type=self.res_type,
    )(
      observations
    )
    return jnp.squeeze(x, -1)


class ResTanhGaussianPolicy(TanhGaussianPolicy):
  hidden_dim: int = 256
  num_blocks: int = 1
  res_type: str = 'v2'

  def setup(self):
    self.base_network = ResNetwork(
      output_dim=self.action_dim,
      hidden_dim=self.hidden_dim,
      num_blocks=self.num_blocks,
      res_type=self.res_type,
    )

    if self.state_dependent_std:
      self.std_network = ResNetwork(
        output_dim=self.action_dim,
        hidden_dim=self.hidden_dim,
        num_blocks=self.num_blocks,
        res_type=self.res_type,
      )
    else:
      self.std = self.param(
        'log_std', nn.initializers.zeros, (self.action_dim,)
      )
      self.std_network = lambda x: self.std

    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)


class ResDirectMappingPolicy(DirectMappingPolicy):
  hidden_dim: int = 256
  num_blocks: int = 1
  res_type: str = 'v2'

  def setup(self):
    self.base_network = ResNetwork(
      self.action_dim,
      hidden_dim=self.hidden_dim,
      num_blocks=self.num_blocks,
      res_type=self.res_type,
    )


class ResClipGaussianPolicy(ClipGaussianPolicy):
  hidden_dim: int = 256
  num_blocks: int = 1
  res_type: str = 'v2'

  def setup(self):
    self.base_network = ResNetwork(
      output_dim=2 * self.action_dim,
      hidden_dim=self.hidden_dim,
      num_blocks=self.num_blocks,
      res_type=self.res_type,
    )
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)
