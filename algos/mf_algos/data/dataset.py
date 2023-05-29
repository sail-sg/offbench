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

"""Dataset utils."""
import os
import rlds
import d4rl
import gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from acme import types # pip install dm-acme
from acme.tf import utils as tf2_utils
from dm_env import StepType
from gym import spaces
from algos.mf_algos.data.rl_unplugged import dm_control_suite


class RandSampler(object):
	"""A random sampler."""

	def __init__(self, max_size: int, batch_size: int = 1) -> None:
		self._max_size = max_size
		self._batch_size = batch_size

	def sample(self):
		"""Return an array of sampled indices."""

		return np.random.randint(self._max_size, size=self._batch_size)


class Dataset(object):
  """Dataset."""

  def __init__(self, data: dict) -> None:
    self._data = data
    self._keys = list(data.keys())
    self._sampler = None

  def size(self):
    return len(self._data[self._keys[0]])

  def retrieve(self, indices: np.ndarray):
    "Get a batch of data."
    indexed = {}

    for key in self._keys:
      indexed[key] = self._data[key][indices, ...]
    return indexed

  @property
  def sampler(self):
    assert self._sampler is not None
    return self._sampler

  def set_sampler(self, sampler):
    self._sampler = sampler

  def sample(self):
    assert self._sampler is not None

    indices = self._sampler.sample()
    return self.retrieve(indices)


class RLUPDataset(object):
  """RL Uplugged dataset."""
  def __init__(self, task_class, task_name, dataset_path, num_threads=8, batch_size=256, num_shards=100, shuffle_buffer_size=100, action_clipping=1, sarsa=True) -> None:
    self._batch_size = batch_size
    self._num_shards = num_shards
    self._shuffle_buffer_size = shuffle_buffer_size
    self._action_clipping = action_clipping
    self._use_sarsa = sarsa

    if task_class == 'control_suite':
      self.task = dm_control_suite.ControlSuite(task_name=task_name)
    elif task_class == 'humanoid':
      self.task = dm_control_suite.CmuThirdParty(task_name=task_name)
    elif task_class == 'rodent':
      self.task = dm_control_suite.Rodent(task_name=task_name)
    else:
      raise NotImplementedError

    assert os.path.isdir(dataset_path)
    _ds = dm_control_suite.dataset(
      root_path=dataset_path,
      data_path=self.task.data_path,
      shapes=self.task.shapes,
      num_threads=num_threads,
      batch_size=self._batch_size,
      uint8_features=self.task.uint8_features,
      num_shards=self._num_shards,
      shuffle_buffer_size=self._shuffle_buffer_size,
    ).prefetch(tf.data.AUTOTUNE)


    def discard_extras(sample):
      return sample._replace(data=sample.data[:5])
    
    def post_proc(data):
      transitions = types.Transition(*data.data)
      observation = transitions.observation
      next_obs = transitions.next_observation
      reward = transitions.reward
      action = transitions.action
      discount = transitions.discount
      next_action = transitions.extras

      observation = tf2_utils.batch_concat(observation)
      next_obs = tf2_utils.batch_concat(next_obs)

      batch = dict(
        observations=observation,
        next_observations=next_obs,
        rewards=reward,
        actions=action,
        next_actions=next_action,
        discounts=discount,
        dones=tf.zeros_like(reward),
      )
      batch['actions'] = tf.clip_by_value(
        batch['actions'], -self._action_clipping, self._action_clipping
      )
      batch['next_actions'] = tf.clip_by_value(
        batch['actions'], -self._action_clipping, self._action_clipping
      )

      return batch

    _ds = _ds.batch(self._batch_size).map(post_proc, num_parallel_calls=tf.data.AUTOTUNE)
    self._ds = iter(_ds)

  def sample(self):
    data = next(self._ds)
    batch = tfds.as_numpy(data)
    return batch

  @property
  def env(self):
    return self.task.environment


class BC_RLUPDataset(object):
  """RL Uplugged dataset."""
  def __init__(self, task_class, task_name, batch_size=256, action_clipping=1, top_n=100) -> None:
    self._batch_size = batch_size
    self._action_clipping = action_clipping

    if task_class == 'control_suite':
      self.task = dm_control_suite.ControlSuite(task_name=task_name)
    elif task_class == 'humanoid':
      self.task = dm_control_suite.CmuThirdParty(task_name=task_name)
    elif task_class == 'rodent':
      self.task = dm_control_suite.Rodent(task_name=task_name)
    else:
      raise NotImplementedError

    def get_traj_return(traj):
      """Calculate trajectory returns for rlup datasets."""
      ret = 0
      for ts in iter(traj[rlds.STEPS]):
        ts = tfds.as_numpy(ts)
        ret += ts[rlds.REWARD]
      return ret
    
    def get_rlup_data(ds):
      """Get observation-action representation for rlup datasets."""
      observations, actions = [], []
      for traj in iter(ds):
        for ts in iter(traj[rlds.STEPS]):
          ts = tfds.as_numpy(ts)
          o = np.concatenate(list(ts[rlds.OBSERVATION].values()))
          a = ts[rlds.ACTION]
          observations.append(o)
          actions.append(a)
      return np.stack(observations), np.stack(actions)
      
    domain = "rlu_control_suite"
    ds_name = f"{domain}/{task_name}"
    path = '../tensorflow_datasets'
    ds = tfds.load(ds_name, data_dir=path, download=False)['train']

    ds = list(iter(ds))
    ds.sort(key=get_traj_return)

    data = {}
    observations, actions = get_rlup_data(ds)
    assert top_n > 0 and top_n <= 100
    new_size = int(top_n/100*observations.shape[0])
  
    data['observations'] = observations[-new_size:]
    data['actions'] = actions[-new_size:]
    data['actions'] = np.clip(
      data['actions'], -self._action_clipping, self._action_clipping
    )
    self._data = data
    self._max_size = new_size
    

  def sample(self):
    indexed = {}
    indices = np.random.randint(self._max_size, size=self._batch_size)
    for key in self._data.keys():
      indexed[key] = self._data[key][indices, ...]
    return indexed

  @property
  def env(self):
    return self.task.environment

  
class DM2Gym(object):
  def __init__(self, env) -> None:
    self._env = env
  
  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return spaces.Box(
      low=action_spec.minimum,
      high=action_spec.maximum,
      shape=action_spec.shape,
      dtype=action_spec.dtype
    )
  
  @property
  def observation_space(self):
    specs = [v for _, v in self._env.observation_spec().items()]
    total_lengths = 0
    for s in specs:
      if len(s.shape) == 0:
        total_lengths += 1
      else:
        total_lengths += s.shape[0]
    return spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(total_lengths,)
    )
  
  def step(self, action):
    ts = self._env.step(action)
    obs = ts.observation
    reward = ts.reward
    done = ts.step_type == StepType.LAST
    info = {}
    obs = self._wrap_obs(obs)
    if not reward:
      reward = 0
    return obs, reward, done, info
  
  def reset(self):
    ts = self._env.reset()
    return self._wrap_obs(ts.observation)
  
  def _wrap_obs(self, obs):
    obs = tf2_utils.add_batch_dim(obs)
    obs = tf2_utils.batch_concat(obs)
    return tfds.as_numpy(obs)
  
  def get_normalized_score(self, reward):
    return reward
