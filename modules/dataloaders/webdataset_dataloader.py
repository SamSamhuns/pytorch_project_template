from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def collate_fn(batch):
    """
    collate function to filter out None
    batch is None when there is an error in loading a data point
    """
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class WebDatasetDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_workers,
                 validation_split=0.,
                 collate_fn=collate_fn,
                 timeout=0,
                 drop_last=False,
                 pin_memory=False,
                 prefetch_factor=2,
                 worker_init_fn=None,
                 persistent_workers=False):
        self.validation_split = validation_split
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        # Note: Recommended to do batching explicitely with WebDataset dataset along with collation
        # instead of using the DataLoader
        self.init_kwargs = {
            'dataset': dataset.batched(batch_size, collation_fn=collate_fn),
            'batch_size': None,
            'collate_fn': None,
            'num_workers': num_workers,
            'timeout': timeout,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'prefetch_factor': prefetch_factor,
            'worker_init_fn': worker_init_fn,
            'persistent_workers': persistent_workers
        }
        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def __len__(self):
        return self.n_samples

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        raise NotImplementedError("""
            Validation split from training dataset not supported for WebDataset DataLoader which
            iters through the webdataset which is an implementation of torch.utils.data.IterableDataset.
            Recommended to create separate train and validation tar datasets instead.
            """)

    def get_validation_split(self):
        if self.valid_sampler is None:
            return None
        else:
            raise NotImplementedError("""
                Validation dataloader creation from train dataset currently not supported.
                Recommended to create separate train and validation tar datasets instead.
                """)
