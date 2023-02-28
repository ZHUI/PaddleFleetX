# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy

import paddle
from paddle.distributed import fleet
import paddle.distributed as dist

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from ppfleetx.utils import config
from ppfleetx.utils.log import logger
from ppfleetx.data import build_dataloader
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine
from ppfleetx.distributed.apis import env


def set_default_flags(flags):
    for flag_name, flag_value in flags.items():
        if os.getenv(flag_name) is None:
            paddle.set_flags({flag_name: flag_value})


if __name__ == "__main__":
    args = config.parse_args()
    cfg = config.get_config(args.config, overrides=args.override, show=False)

    paddle.set_device(cfg["Global"]["device"])
    if dist.get_world_size() > 1:
        env.init_dist_env(cfg)

    env.set_seed(cfg.Global.seed)

    module = build_module(cfg)
    from distutils.util import strtobool
    import os
    load_torch_random_345m_ckpt = strtobool(
        os.getenv("load_torch_random_345m_ckpt", False))
    if load_torch_random_345m_ckpt:
        from load_t2p import trans
        import torch
        torch_state = torch.load(
            '/root/paddlejob/workspace/env_run/gpt_benchmark/Megatron-LM/ckpt_345m_init_fp32.bin',
            map_location="cpu")
        paddle_state = trans(torch_state)
        missing_keys, unexpected_keys = module.model.set_state_dict(
            paddle_state)
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)

    config.print_config(cfg)

    train_data_loader = build_dataloader(cfg.Data, "Train")
    eval_data_loader = build_dataloader(cfg.Data, "Eval")

    cfg.Optimizer.lr.update({
        'epochs': cfg.Engine.num_train_epochs,
        'step_each_epoch': len(train_data_loader),
        'total_steps': cfg.Engine.max_steps,
    })

    engine = EagerEngine(configs=cfg, module=module)

    if cfg.Engine.save_load.ckpt_dir is not None:
        engine.load()

    engine.fit(train_data_loader=train_data_loader,
               valid_data_loader=eval_data_loader,
               epoch=cfg.Engine.num_train_epochs)
