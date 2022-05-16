from torch.utils.data import Dataset


def get_dataset(dataset_name: str) -> Dataset:
    if dataset_name == 'lizard':
        return cbmi_utils.pytorch.datasets.lizard.LizardDetection