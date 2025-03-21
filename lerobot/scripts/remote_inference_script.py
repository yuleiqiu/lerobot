#!/usr/bin/env python

import logging
import pickle
import sys
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
import matplotlib.pyplot as plt

import einops
import numpy as np
import torch
from termcolor import colored

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


@parser.wrap()
def inference(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    obs_file =
    action_file =

    with open(obs_file, 'rb') as f:
        observation = pickle.load(f)

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, local_files_only=cfg.dataset.local_files_only)
    logging.info("Making policy.")
    policy = make_policy(
        cfg=cfg.policy,
        device=device,
        ds_meta=ds_meta,
    )
    policy.eval()

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext():
        action = policy.select_action(observation)

    if isinstance(action, torch.Tensor):
        action = action.squeeze(0).cpu().numpy()

    with open(action_file, 'wb') as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    init_logging()
    eval()