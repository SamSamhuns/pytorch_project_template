
import torch
from torch.utils.data import Dataset


# Dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, samples: int, image_size: tuple[int, int], seed: int = 42):
        self.samples = samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.samples:
            raise IndexError(f"Index {idx} is out of range for dataset of size {self.samples}.")
        # Use a fixed seed for reproducibility
        torch.manual_seed(idx + self.seed)
        # Generate random RGB image of specified size
        image = torch.rand(3, *self.image_size)
        label = torch.tensor(idx)  # Dummy label
        return image, label
