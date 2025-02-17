"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType


import pdb


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/real_aloha/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 8000
    log_freq = 100

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("yqiu777/aloha_mobile_left_dom", local_files_only=True)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = ACTConfig(input_features=input_features,
                    output_features=output_features,
                    chunk_size=50,
                    n_action_steps=1,
                    temporal_ensemble_coeff=0.01,
                    n_decoder_layers=7,
                    optimizer_lr=5e-5)

    # We can now instantiate our policy with this config and the dataset stats.
    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # We can then instantiate the dataset.
    dataset = LeRobotDataset("yqiu777/aloha_mobile_left_dom", local_files_only=True)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=32,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    for batch in dataloader:
        print("Batch keys:", batch.keys())
        for key, value in batch.items():
            print(f"Key: {key} -> Type: {type(value)}", end='')
            # 如果是 tensor，可以打印其 shape
            if hasattr(value, "shape"):
                print(f", Shape: {value.shape}")
            else:
                print()
        break

    # # Run training loop.
    # step = 0
    # done = False
    # while not done:
    #     for raw_batch in dataloader:
    #         input_batch = {k: v for k, v in raw_batch.items() if k == "action" or k.startswith("observation")}
    #         input_batch["action"] = input_batch["action"].unsqueeze(1).repeat(1, 50, 1) # chunk_size
    #         for key, value in input_batch.items():
    #             print(f"Key: {key} -> Type: {type(value)}", end='\n')
    #         input_batch = {k: v.to(device, non_blocking=True) for k, v in input_batch.items()}
    #         loss, loss_dict = policy.forward(input_batch)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()

    #         if step % log_freq == 0:
    #             print(f"step: {step} loss: {loss.item():.3f}, l1_loss: {loss_dict['l1_loss']:.3f}, kld_loss: {loss_dict['kld_loss']:.3f}")
    #         step += 1
    #         break
            # if step >= training_steps:
            #     done = True
            #     break

    # # Save a policy checkpoint.
    # policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
