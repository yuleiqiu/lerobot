import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


def main():
    dataset = LeRobotDataset("yqiu777/aloha_mobile_left_dom", local_files_only=True)
    dataset.push_to_hub()

if __name__ == "__main__":
    main()