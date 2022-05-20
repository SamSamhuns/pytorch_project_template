#  splits a directory with object classes in different subdirectories into
#  train, test and optionally val sub-directory with the same class sub-dir
#  structure

import os
import glob
import shutil
import random
import argparse
from tqdm import tqdm
from typing import List
from util import _fix_path_for_globbing

# #################### Raw Data Organization #########################
#   raw_data
#          |_ dataset
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

# #################### Data Configurations here #####################
# example raw data path = "data/raw_data/birds_dataset"
# example target data path = "data/processed_birds_dataset"
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm'}
random.seed(42)
# ###################################################################


def main():
    """By default dataset is spit in train-val-test in ratio 80:10:10.
    If one of the val or test split is not provided,
    data is split into two parts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-rd',
                        '--raw_data_path',
                        type=str,
                        required=True,
                        help="""Raw dataset path with
                        class imgs inside folders""")
    parser.add_argument('-td',
                        '--target_data_path',
                        type=str,
                        required=True,
                        help="""Target dataset path where
                        imgs will be sep into train, val or test""")
    parser.add_argument('-vs',
                        '--val_split',
                        type=float,
                        required=False,
                        help='Val data split percentage. i.e. 0.1')
    parser.add_argument('-ts',
                        '--test_split',
                        type=float,
                        required=False,
                        help='Test data split percentage. i.e. 0.1')
    args = parser.parse_args()
    # set default vals here instead of above to ensure None
    # vals will be passed if only one split is provided
    if args.val_split is None and args.test_split is None:
        args.val_split = 0.10
        args.test_split = 0.10

    split_train_test(args.raw_data_path,
                     args.target_data_path,
                     args.val_split,
                     args.test_split)


def create_dir_and_copy_files(dir: str, f_list: List[str]) -> None:
    """dir: directory where files will be copied to
    f_list: list of files which will be copied to dir
    """
    os.makedirs(dir, exist_ok=True)
    for file in f_list:
        shutil.copy(file, dir)


def split_train_test(RAW_IMG_DIR, PROCESSED_IMG_DIR, VAL_SPLIT, TEST_SPLIT) -> None:
    train_dir = os.path.join(PROCESSED_IMG_DIR, "train")
    os.makedirs(train_dir, exist_ok=True)

    if VAL_SPLIT is not None:
        val_dir = os.path.join(PROCESSED_IMG_DIR, "val")
        os.makedirs(val_dir, exist_ok=True)
    if TEST_SPLIT is not None:
        test_dir = os.path.join(PROCESSED_IMG_DIR, "test")
        os.makedirs(test_dir, exist_ok=True)

    dir_list = glob.glob(_fix_path_for_globbing(RAW_IMG_DIR))

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        dir = dir_list[i]                # get path to class dir
        class_name = dir.split("/")[-1]  # get class name

        f_list = [file for file in glob.glob(dir + "/*")
                  if file.split(".")[-1] in VALID_FILE_EXTS]
        random.shuffle(f_list)

        train = []
        val_size, test_size = 0, 0
        if VAL_SPLIT is not None:
            # get val size
            val_size = int(len(f_list) * VAL_SPLIT)
            val = [f_list[i] for i in range(val_size)]
            class_val_dir = os.path.join(val_dir, class_name)
            create_dir_and_copy_files(class_val_dir, val)
        if TEST_SPLIT is not None:
            # get test size
            test_size = int(len(f_list) * TEST_SPLIT)
            test = [f_list[i + val_size] for i in range(test_size)]
            class_test_dir = os.path.join(test_dir, class_name)
            create_dir_and_copy_files(class_test_dir, test)

        train = [f_list[val_size + test_size + i]
                 for i in range(len(f_list) - (val_size + test_size))]
        class_train_dir = os.path.join(train_dir, class_name)
        create_dir_and_copy_files(class_train_dir, train)


if __name__ == "__main__":
    main()
