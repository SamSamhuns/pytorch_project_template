"""Duplicate data"""
import os
import os.path as osp
import glob
import shutil
import argparse

from tqdm import tqdm


# #################### Raw Data Organization ########################
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

# #################### Data configurations here #####################
VALID_FILE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm'}
# ###################################################################


def safe_copy(file_path, out_dir, dst=None) -> None:
    """Safely copy a file to the specified directory.
    If a file with the same name already
    exists, the copied file name is altered to preserve both.

    :param str file_path: Path to the file to copy.
    :param str out_dir: Directory to copy the file into.
    :param str dst: New name for the copied file.
    If None, use the name of the original file.
    """
    name = dst or osp.basename(file_path)
    if not osp.exists(osp.join(out_dir, name)):
        shutil.copy(file_path, osp.join(out_dir, name))
    else:
        base, extension = osp.splitext(name)
        i = 1
        while osp.exists(osp.join(out_dir, '{}_{}{}'.format(base, i, extension))):
            i += 1
        shutil.copy(file_path, osp.join(
            out_dir, '{}_{}{}'.format(base, i, extension)))


def split_train_test(RAW_IMG_DIR, DUPLICATED_IMG_DIR, TARGET_NUMBER) -> None:
    target_dir = DUPLICATED_IMG_DIR
    os.makedirs(target_dir, exist_ok=True)

    # fix path for globbing
    if not RAW_IMG_DIR.endswith(('/', '*')):
        RAW_IMG_DIR += '/'
    dir_list = glob.glob(RAW_IMG_DIR + '*')

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        directory = dir_list[i]                # get path to class directory
        class_name = directory.split("/")[-1]  # get class name
        f_list = [file for file in sorted(glob.glob(directory + "/*"))
                  if file.split(".")[-1] in VALID_FILE_EXTS]

        class_target_dir = osp.join(target_dir, class_name)

        # skip copying if directory already exists and has required num of files
        if osp.exists(class_target_dir):
            tf_list = glob.glob(class_target_dir + "/*")
            if len(tf_list) >= TARGET_NUMBER:
                continue
        os.makedirs(class_target_dir, exist_ok=True)
        f_count = 0

        # iterate through all files & copy till target num is reached
        while f_count <= TARGET_NUMBER:
            for file in f_list:
                safe_copy(file, class_target_dir)
                f_count += 1
                if f_count > TARGET_NUMBER:
                    break
            if f_count == 0:  # no files in directory
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--raw_data_path",
                        type=str, dest="raw_data_path",
                        required=True,
                        help="""Raw dataset path with class imgs inside folders""")
    parser.add_argument("-td", "--target_data_path",
                        type=str, dest="target_data_path",
                        required=True,
                        help="""Target dataset path where imgs will be saved in sub folders
                        repr classes with number matching target_number""")
    parser.add_argument("-n", "--target_number",
                        type=int,
                        required=False,
                        default=1000,
                        help="""Target size to reach for
                        each class after duplication (default: %(default)s)""")
    args = parser.parse_args()
    split_train_test(args.raw_data_path,
                     args.target_data_path,
                     args.target_number)


if __name__ == "__main__":
    main()
