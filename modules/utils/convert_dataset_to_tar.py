# convert dataset to tar format for fast loading with webdataset
import glob
import argparse
from tqdm import tqdm
import webdataset as wds
import imageio.v2 as imageio
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
                 tar_path: str,
                 mapping_fname: str = 'dataset_mapping.txt') -> None:
    """ generates a combined tar archive for loading into webdataset
    from class folder separated data from src_data_dir & a class mapping txt file
    """
    dir_list = glob.glob(_fix_path_for_globbing(src_data_dir))
    class_id = 0
    file_count = 1

    with open(mapping_fname, 'w', encoding="utf-8") as map_file, wds.TarWriter(tar_path) as sink:
        for dir_name in tqdm(dir_list):
            split_string = dir_name.split('/')
            map_file.write(str(class_id) + "\t" + split_string[-1] + "\n")
            img_list = glob.glob(dir_name + "/*")
            for img_name in img_list:
                try:
                    img = imageio.imread(img_name, pilmode="RGB")
                    img = img[..., :3]  # drop alpha chanbel if it exists
                    assert img.ndim == 3 and img.shape[2] == 3
                    assert isinstance(class_id, int)

                    sink.write({
                        "__key__": "sample%06d" % file_count,
                        "input.jpg": img,
                        "output.cls": class_id,
                    })
                    file_count += 1
                except Exception as excep:
                    print(
                        f"{excep}. imageio could not read file {img_name}. Skipping...")
            class_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", "--source_data_path",
                        type=str, dest="source_data_path",
                        required=True,
                        help="""Source dataset path with
                        class imgs inside folders""")
    parser.add_argument("--td", "--target_tar_path",
                        type=str, dest="target_tar_path",
                        required=True,
                        help="""Target tar path where the
                        data is stored for fast retrieval i.e. data/custom_data.tar""")
    parser.add_argument("--mp", "--mapping_file_path",
                        type=str, dest="mapping_file_path",
                        required=True,
                        help="""Mapping file txt path where
                        class names and index ids will be present""")
    args = parser.parse_args()
    generate_tar(args.source_data_path,
                 args.target_tar_path,
                 args.mapping_file_path)


if __name__ == "__main__":
    main()
