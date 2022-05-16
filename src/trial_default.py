from typing import Union, Dict, Any

import torch
from determined import pytorch
from determined.pytorch import DataLoader, PyTorchTrial

import datasets


class DefaultTrial(PyTorchTrial):
    def __init__(self, context):
        super().__init__(context)
        self.context = context

        self.dataset_name = self.context.get_hparam('dataset')
        self.num_workers = self.context.get_hparam('num_workers')

    def train_batch(
            self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        pass

    def build_training_data_loader(self) -> pytorch.DataLoader:
        dataset = datasets.get_dataset(self.dataset_name)
        # TODO: Augmentation (Flip, Turn 90 Degrees)
        dataloader = DataLoader(
            dataset,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return dataloader

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        pass
