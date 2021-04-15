import os
import glob
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from util import _fix_path_for_globbing

##################### Raw Data Organization ########################
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
#
#   npz_data
#          |_  000001.npz
#          |_  000001.npz
#          |_  000001.npz
#          ...
####################################################################


def generate_npz_files(raw_img_path,
                       npz_path,
                       mapping_fname='dataset_mapping.txt') -> Noneh:
    """ generates a flattened list of npz files
    from class folder separated data from raw_img_path
    """
    os.makedirs(npz_path, exist_ok=True)
    dir_list = glob.glob(_fix_path_for_globbing(raw_img_path))
    class_id = 0
    npz_name = 1

    with open(mapping_fname, 'w') as map_file:
        for i in tqdm(range(len(dir_list))):
            dir_name = dir_list[i]
            split_string = dir_name.split('/')
            map_file.write(str(class_id) + "\t" + split_string[-1] + "\n")
            img_list = glob.glob(dir_name + "/*")
            for img_name in img_list:
                # print(img_name)
                try:
                    img = imageio.imread(img_name, pilmode="RGB")
                    img = img[..., :3]
                    np.savez(os.path.join(npz_path, str(npz_name).zfill(6)),
                             image=img, class_id=class_id)
                    npz_name += 1
                except Exception as e:
                    print(f"{e}. imageio could not read file {img_name}")
            class_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd',
                        '--source_data_path',
                        type=str,
                        required=True,
                        help="""Source dataset path with
                        class imgs inside folders""")
    parser.add_argument('-td',
                        '--target_npz_path',
                        type=str,
                        required=True,
                        help="""Target dataset path where the
                        dir structure is flattened & imgs saved as npz""")
    parser.add_argument('-m',
                        '--mapping_file_path',
                        type=str,
                        required=True,
                        help="""Mapping file txt path where
                        class names and index ids will be present""")
    args = parser.parse_args()
    generate_npz_files(args.source_data_path,
                       args.target_npz_path,
                       args.mapping_file_path)


if __name__ == "__main__":
    main()
