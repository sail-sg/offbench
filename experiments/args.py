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

import os

from ml_collections import ConfigDict, config_flags

import algos
from utilities.utils import WandBLogger, define_flags_with_default

algo_cfg_default_collection = ConfigDict()
for alg in algos.__all__:
  algo_cfg_default_collection.update(getattr(algos, alg).get_default_config())

release = False
if 'RELEASE' in os.environ:
  env_rel = os.environ['RELEASE']
  if env_rel == 'true':
    release = True

FLAGS_DEF = define_flags_with_default(
  env="walker2d-medium-v2",
  dataset='d4rl',
  rl_unplugged_task_class='control_suite',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=100,
  logging=WandBLogger.get_default_config(),
  release=release,
  # configs for dataset distribution
  topn=100, # use top x% transitions sorted by traj return
)

config_flags.DEFINE_config_file(
  'algo_cfg',
  help_string='algorithm configuration file',
  lock_config=False,
)
