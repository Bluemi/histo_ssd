from typing import Union, Dict, Any

import torch
from determined import pytorch
from determined.pytorch import PyTorchTrial


class DefaultTrial(PyTorchTrial):
    def __init__(self, context):
        super().__init__(context)
        self.context = context

        self.dataset_name = self.context.get_hparam('dataset')

    def train_batch(
            self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        pass

    def build_training_data_loader(self) -> pytorch.DataLoader:
        pass

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        pass
