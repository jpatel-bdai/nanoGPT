import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union
import lightning as L
import numpy as np
import os

from lightning_scripts import config

torch.manual_seed(0)


class OpenWebTextDataset(Dataset):
    def __init__(self, data: str, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return config.BATCH_SIZE * config.MAX_ITERS * config.DEVICES

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (1,))
        x = torch.from_numpy((self.data[ix : ix + self.block_size]).astype(np.int64))
        y = torch.from_numpy(
            (self.data[ix + 1 : ix + 1 + self.block_size]).astype(np.int64)
        )
        return x, y


class GPTDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, block_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_data = np.memmap(
            os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        self.val_data = np.memmap(
            os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        self.train_dataset = OpenWebTextDataset(self.train_data, self.block_size)
        self.val_dataset = OpenWebTextDataset(self.val_data, self.block_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
