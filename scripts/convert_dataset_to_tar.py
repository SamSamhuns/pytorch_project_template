"""Convert one split of an image folder dataset to WebDataset tar shards for streaming large datasets.
The images are stored as they are, without re-encoding, and in shuffled order so that each shard mixes the classes.
info.json next to the shards holds the class names and the image count, which training reads.
"""

import argparse
import json
import os
import os.path as osp
import random

import webdataset as wds
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# #################### Raw Data Organization ########################
#   raw_data
#          |_ dataset_split
#                   |_ class_1
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   |_ class_2
#                             |_ img1
#                             |_ img2
#                             |_ ....
#                   ...
# ###################################################################


def write_shards(src_data_dir: str, target_dir: str, maxcount: int = 1000) -> None:
    """Write the images of src_data_dir to target_dir/000000.tar, 000001.tar, ... with maxcount images each,
    and their class names and count to target_dir/info.json
    """
    dataset = ImageFolder(src_data_dir)  # the same classes and labels as data_mode: imgs
    samples = dataset.samples.copy()
    random.Random(42).shuffle(samples)  # a fixed seed, so the same images always give the same shards
    os.makedirs(target_dir)  # raises if it exists, as old shards there would be read too

    with wds.ShardWriter(osp.join(target_dir, "%06d.tar"), maxcount=maxcount) as sink:
        for index, (path, class_idx) in enumerate(tqdm(samples)):
            # the file extension, e.g. jpg or png, tells the loader how to decode the bytes
            extension = osp.splitext(path)[1][1:].lower()
            with open(path, "rb") as image_file:
                sink.write({"__key__": f"{index:08d}", extension: image_file.read(), "cls": class_idx})

    with open(osp.join(target_dir, "info.json"), "w", encoding="utf-8") as info_file:
        json.dump({"classes": dataset.classes, "num_samples": len(samples)}, info_file, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sd",
        "--source_data_path",
        type=str,
        dest="source_data_path",
        required=True,
        help="Source split with one sub-folder of images per class, e.g. data/birds_dataset/train",
    )
    parser.add_argument(
        "--td",
        "--target_dir",
        type=str,
        dest="target_dir",
        required=True,
        help="New dir for the shards and info.json, e.g. data/birds_webdataset/train",
    )
    parser.add_argument(
        "--maxcount",
        type=int,
        default=1000,
        help="Images per shard. Each DataLoader worker reads its own shards, so write at least as many shards "
        "as dataloader.num_workers (default: %(default)s)",
    )
    args = parser.parse_args()
    write_shards(args.source_data_path, args.target_dir, args.maxcount)


if __name__ == "__main__":
    main()
