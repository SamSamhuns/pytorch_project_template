import pytest
from torchvision import transforms
from src.datasets.base_dataset import ImageFolderDataset


# ######### Fixtures for ucr_clsf_dataset.CustomUCR #########


@pytest.fixture()
def image_dataset(mock_img_data_dir):
    """Initialize the dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return ImageFolderDataset(str(mock_img_data_dir), transform=transform)
