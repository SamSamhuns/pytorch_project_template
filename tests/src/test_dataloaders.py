from torch.utils.data import DataLoader
import torch
import pytest

from src.dataloaders import CustomDataLoader, WebDatasetDataLoader
from src.dataloaders.webdataset_dataloader import collate_fn
from tests.conftest import NUM_CLS, NUM_IMGS_P_CLS


class TestCustomDataLoader:

    def test_initialization(self, mock_dataset):
        """Tests for CustomDataLoader"""
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=1)
        for b in loader:
            assert b.shape == (9, 10)
        assert loader.n_samples == 100

    def test_validation_split(self, mock_dataset):
        """Test with + without validation split"""
        # Test without validation_split
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=1, validation_split=0)
        assert loader.valid_sampler is None
        assert len(loader.sampler) == 100

        # Test with validation split
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=1, validation_split=0.2)
        assert len(loader.valid_sampler) == 20  # 20% of 100
        assert len(loader.train_sampler) == 80  # remaining 80%

    def test_collate_fn(self, mock_dataset):
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=1)
        for batch in loader:
            # Check that no None values are in the batch
            # since default collate should remove None
            assert not any(x is None for x in batch)

    def test_no_workers(self, mock_dataset):
        """Test with num_workers=0"""
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=0, prefetch_factor=None)
        for batch in loader:
            # Last batch might be smaller if not dropping the last
            assert len(batch) == 9

    def test_persistent_workers(self, mock_dataset):
        """This will test the scenario where persistent workers are set to True"""
        loader = CustomDataLoader(
            mock_dataset, batch_size=10, shuffle=False, num_workers=1, persistent_workers=True)
        total_batches = sum(1 for _ in loader)
        # There should be 9 batches of 10 items each (last batch is dropped)
        assert total_batches == 10


class TestWebDatasetDataLoader:
    def test_initialization(self, dummy_webdataset):
        loader = WebDatasetDataLoader(
            dummy_webdataset, batch_size=4, num_workers=0, prefetch_factor=None)
        assert loader is not None
        assert loader.n_samples == NUM_CLS * NUM_IMGS_P_CLS

    def test_collate_fn_output(self):
        # Test with a mixed list of tensors and None values
        batch = [torch.tensor([1, 2]), None, torch.tensor([3, 4])]
        result = collate_fn(batch)
        assert len(result) == 2  # Only two tensors should be returned

    def test_validation_split_error(self, dummy_webdataset):
        with pytest.raises(NotImplementedError):
            _ = WebDatasetDataLoader(
                dummy_webdataset, batch_size=4, num_workers=0, validation_split=0.1)
