# convert dataset to tar format for fast loading with webdataset
import glob
import argparse
from tqdm import tqdm
from modules.utils.common import _fix_path_for_globbing


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


def generate_tar(src_data_dir: str,
                 mapping_file_path: str = 'dataset_mapping.txt') -> None:
    """ generates a class id2name mapping txt file based on the 1st level directory structure
    """
    dir_list = sorted(glob.glob(_fix_path_for_globbing(src_data_dir)))
    class_id = 0

    with open(mapping_file_path, "w", encoding="utf-8") as mptr:
        for dir_path in tqdm(dir_list):
            dir_name = dir_path.split('/')[-1]
            mptr.write(f"{class_id}\t{dir_name}\n")
            class_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", "--source_data_path",
                        type=str, dest="source_data_path",
                        required=True,
                        help="""Source dataset path with
                        class imgs inside folders""")
    parser.add_argument("--mp", "--mapping_file_path",
                        type=str, dest="mapping_file_path",
                        required=True,
                        help="""Mapping file txt path where
                        class names and index ids will be present""")
    args = parser.parse_args()
    generate_tar(args.source_data_path,
                 args.mapping_file_path)


if __name__ == "__main__":
    main()
