#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Callable

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.eval import EvalPipelineConfig


def rollout(
    env: RealEnv,
    policy: PreTrainedPolicy,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A a dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE the that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    ts = env.reset()

    # all_states = []
    # all_images = []
    # all_actions = []
    all_rewards = []
    # all_successes = []

    max_timesteps = int(max_timesteps * 2) # may increase for real-world tasks

    with torch.inference_mode():
        time0 = time.time()
        DT = 1 / FPS
        culmulated_delay = 0
        for t in range(max_timesteps):
            time1 = time.time()

            observation = ts.observation
            observation = preprocess_observation(observation)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

            if t == 0:
                # warm up
                for _ in range(10):
                    policy.select_action(observation)
                print('network warm up done')
                time1 = time.time()
            
            ### query policy
            if t % policy.n_action_steps == 0:
                action = policy.select_action(observation)

                # Convert to CPU / numpy.
                action = action.squeeze(0).to("cpu").numpy()
                assert action.ndim == 1, "Action dimensions should be (action_dim, )"

            target_qpos = action
            base_action = np.array([0.0, 0.0])

            # step the environment
            ts = env.step(target_qpos, base_action)

            ### for visualization
            # qpos_list.append(qpos_numpy)
            # target_qpos_list.append(target_qpos)
            all_rewards.append(ts.reward)
            duration = time.time() - time1
            sleep_time = max(0, DT - duration)
            time.sleep(sleep_time)
            if duration >= DT:
                culmulated_delay += (duration - DT)
                print((
                    f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: '
                    f'{DT} s, culmulated delay: {culmulated_delay:.3f} s'
                ))
        print(f'Avg fps {max_timesteps / (time.time() - time0)}')

    from aloha.robot_utils import move_grippers # requires aloha
    move_grippers(
        [env.follower_bot_left, env.follower_bot_right],
        [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
        moving_time=0.5,
    )  # open

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return


def eval_policy(
    env: RealEnv,
    policy: PreTrainedPolicy,
    num_rollouts: int,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # # Keep track of some metrics.
    # sum_rewards = []
    # max_rewards = []
    # all_successes = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(num_rollouts, desc="Stepping through eval rollouts", disable=inside_slurm())
    for rollout_id in progbar:
        rollout(
            env,
            policy,
            # return_observations=return_episode_data,
        )

        # # FIXME: episode_data is either None or it doesn't exist
        # if return_episode_data:
        #     this_episode_data = _compile_episode_data(
        #         rollout_data,
        #         done_indices,
        #         start_episode_index=batch_ix * env.num_envs,
        #         start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
        #         fps=env.unwrapped.metadata["render_fps"],
        #     )
        #     if episode_data is None:
        #         episode_data = this_episode_data
        #     else:
        #         # Some sanity checks to make sure we are correctly compiling the data.
        #         assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
        #         assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
        #         # Concatenate the episode data.
        #         episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}


        progbar.set_postfix(
            {f"Rollout {rollout_id}\n"}
            # {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # # Compile eval info.
    # info = {
    #     "per_episode": [
    #         {
    #             "episode_ix": i,
    #             "sum_reward": sum_reward,
    #             "max_reward": max_reward,
    #             "success": success,
    #             "seed": seed,
    #         }
    #         for i, (sum_reward, max_reward, success, seed) in enumerate(
    #             zip(
    #                 sum_rewards[:n_episodes],
    #                 max_rewards[:n_episodes],
    #                 all_successes[:n_episodes],
    #                 all_seeds[:n_episodes],
    #                 strict=True,
    #             )
    #         )
    #     ],
    #     "aggregated": {
    #         "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
    #         "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
    #         "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
    #         "eval_s": time.time() - start,
    #         "eval_ep_s": (time.time() - start) / n_episodes,
    #     },
    # }

    # if return_episode_data:
    #     info["episodes"] = episode_data

    # return info
    return


@parser.wrap()
# def eval(cfg: EvalPipelineConfig):
def eval(cfg: TrainPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    from aloha.real_env import make_real_env # requires aloha
    from aloha.robot_utils import move_grippers # requires aloha
    from interbotix_common_modules.common_robot.robot import (
        create_interbotix_global_node,
        get_interbotix_global_node,
        robot_startup,
    )
    from interbotix_common_modules.common_robot.exceptions import InterbotixException
    try:
        node = get_interbotix_global_node()
    except:
        node = create_interbotix_global_node('aloha')
    logging.info("Making environment.")
    env = make_real_env(node=node, setup_robots=True, setup_base=True)
    try:
        robot_startup(node)
    except InterbotixException:
        pass

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        ds_meta=dataset.meta,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
        # info = eval_policy(
        eval_policy(
            env,
            policy,
            cfg.eval.n_episodes,
        )
    # print(info["aggregated"])

    # # Save info
    # with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
    #     json.dump(info, f, indent=2)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval()
