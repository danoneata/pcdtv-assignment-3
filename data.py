from typing import Literal, TypedDict

import torch
from speechbrain.dataio.batch import PaddedBatch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import VoxCeleb1Identification

Split = Literal["train", "valid", "test"]
Sample = TypedDict("Sample", {"feature": torch.tensor, "identity": int})


class VoxCeleb1Features(Dataset):
    def __init__(self, root: str, n_mels:int, split: Split):
        self.voxceleb1 = VoxCeleb1Identification(
            subset="test",
            root=root,
            download=False,
        )
        raise NotImplementedError

    def __getitem__(self, i: int) -> Sample:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def setup_data(config):
    n_mels = config.features.n_mels
    root = config.data_dir
    train_dataset = VoxCeleb1Features(root, n_mels, "train")
    valid_dataset = VoxCeleb1Features(root, n_mels, "valid")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        num_workers=config.num_workers,
        collate_fn=PaddedBatch,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size_valid,
        num_workers=config.num_workers,
        collate_fn=PaddedBatch,
        shuffle=False,
    )

    return train_dataloader, valid_dataloader
