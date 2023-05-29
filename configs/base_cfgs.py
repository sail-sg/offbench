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

import ml_collections

def get_default_training_cfgs():
  config = ml_collections.ConfigDict()

  config.batch_size = 256
  config.policy_layer_norm = False
  config.qf_layer_norm = False
  config.activation = "elu"
  config.policy_arch="256-256"
  config.qf_arch="256-256"
  config.obs_norm = False
  config.state_dependent_std = True
  config.tanh_squash_distribution = True
  config.norm_reward = False
  config.use_resnet = False
  config.hidden_dim = 256
  config.res_type = 'crr'
  config.num_blocks = 4
  config.clip_mean = True
  config.orthogonal_init = False
  config.n_epochs=1200

  config.dropout = None
  config.rew_clip = False
  config.target_policy_warmup = 0
  config.target_policy_update_interval = 1
  config.last_layer_init = 0.01
  config.note = ''

  return config
