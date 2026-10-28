"""Copy each class dir's images into target dirs, repeating them until each class has target_number images"""

import argparse
import glob
import os
import os.path as osp
import shutil

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
VALID_FILE_EXTS = {"jpg", "jpeg", "png", "ppm", "bmp", "pgm"}
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
        while osp.exists(osp.join(out_dir, f"{base}_{i}{extension}")):
            i += 1
        shutil.copy(file_path, osp.join(out_dir, f"{base}_{i}{extension}"))


def duplicate_data(raw_img_dir: str, target_dir: str, target_number: int) -> None:
    os.makedirs(target_dir, exist_ok=True)

    # fix path for globbing
    if not raw_img_dir.endswith(("/", "*")):
        raw_img_dir += "/"
    dir_list = glob.glob(raw_img_dir + "*")

    # for each class in raw data
    for i in tqdm(range(len(dir_list))):
        directory = dir_list[i]  # get path to class directory
        class_name = directory.split("/")[-1]  # get class name
        f_list = [
            file for file in sorted(glob.glob(directory + "/*")) if file.split(".")[-1].lower() in VALID_FILE_EXTS
        ]

        class_target_dir = osp.join(target_dir, class_name)
        os.makedirs(class_target_dir, exist_ok=True)

        # count the images already there, e.g. when the target dir is the raw dir, and copy until there are enough
        f_count = len(glob.glob(class_target_dir + "/*"))
        while f_count < target_number and f_list:
            for file in f_list[: target_number - f_count]:
                safe_copy(file, class_target_dir)
                f_count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rd",
        "--raw_data_path",
        type=str,
        dest="raw_data_path",
        required=True,
        help="""Raw dataset path with class imgs inside folders""",
    )
    parser.add_argument(
        "--td",
        "--target_data_path",
        type=str,
        dest="target_data_path",
        required=True,
        help="""Target dataset path where imgs will be saved in sub folders
                        repr classes with number matching target_number""",
    )
    parser.add_argument(
        "-n",
        "--target_number",
        type=int,
        required=False,
        default=1000,
        help="""Target size to reach for
                        each class after duplication (default: %(default)s)""",
    )
    args = parser.parse_args()
    duplicate_data(args.raw_data_path, args.target_data_path, args.target_number)


if __name__ == "__main__":
    main()
