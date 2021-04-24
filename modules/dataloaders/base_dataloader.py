import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers,
                 collate_fn=default_collate,
                 timeout=0,
                 drop_last=False,
                 pin_memory=False,
                 prefetch_factor=2,
                 worker_init_fn=None,
                 persistent_workers=False):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'timeout': timeout,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor,
            'worker_init_fn': worker_init_fn,
            'persistent_workers': persistent_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


if __name__ == "__main__":
    # for image datasets with input data organized within separate folders
    # based on their parent labels use torchvision.datasets.ImageFolder
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    dataset = datasets.ImageFolder(root="data/birds_dataset/valid",
                                   transform=transforms.ToTensor())
    print(dataset)
    print(f"len of dataset(Number of datapoints): {len(dataset)}")
    data_loader = BaseDataLoader(dataset,
                                 batch_size=32,
                                 shuffle=True,
                                 validation_split=0.1,
                                 num_workers=2)

    # train dataloader
    train_len = 0
    for (X_batch, y_batch) in data_loader:
        X, y = X_batch, y_batch
        train_len += X.shape[0]
    print("train dataloader")
    print(f"\t shape of last batch(X,y): {X.shape, y.shape}")
    print(f"\t num of datapoints: {train_len}")

    # validation dataloader
    if data_loader.split_validation() is not None:
        val_len = 0
        for (X_batch, y_batch) in data_loader.split_validation():
            X, y = X_batch, y_batch
            val_len += X.shape[0]
        print("val dataloader")
        print(f"\t shape of last batch(X,y): {X.shape, y.shape}")
        print(f"\t num of datapoints: {val_len}")
