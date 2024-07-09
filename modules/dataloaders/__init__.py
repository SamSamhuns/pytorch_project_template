from .base_dataloader import CustomDataLoader
from .webdataset_dataloader import WebDatasetDataLoader


IMPLEMENTED_DATALOADERS = {
    "CustomDataLoader": CustomDataLoader,
    "WebDatasetDataLoader": WebDatasetDataLoader}


def init_dataloader(dataloader_name: str, **kwargs):
    """Initialize the dataloader."""
    try:
        dataloader = IMPLEMENTED_DATALOADERS[dataloader_name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"{dataloader_name} is not implemented. " +
            f"Available Dataloaders: {IMPLEMENTED_DATALOADERS.keys()}") from exc
    return dataloader
