import os
import glob
import imageio
import argparse
from tqdm import tqdm
from util import _fix_path_for_globbing


def validate_imgs(source_path, corrupt_flist_txt, remove) -> None:
    """ validate imgs and optionally remove corrupt images
    """
    dir_list = glob.glob(_fix_path_for_globbing(source_path))

    with open(corrupt_flist_txt, 'w') as fw:
        for i in tqdm(range(len(dir_list))):
            dir_name = dir_list[i]
            img_list = glob.glob(dir_name + "/*")
            for img_name in img_list:
                # print(img_name)
                try:
                    img = imageio.imread(img_name, pilmode="RGB")
                    img = img[..., :3]
                except Exception as e:
                    print(f"{e}. imageio could not read file {img_name}")
                    fw.write(img_name + '\n')
                    if os.path.exists(img_name) and remove:
                        print(f"Removing {img_name}")
                        os.remove(img_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--source_data_path',
                        type=str,
                        required=True,
                        help="""Source dataset path with
                        class imgs inside folders""")
    parser.add_argument('-r',
                        '--remove',
                        type=bool,
                        default=False,
                        required=False,
                        help="""Remove corrupt imgs. By default,
                        the imgs are only listed and saved in a file""")
    parser.add_argument('-t',
                        '--corrupt_file_list_txt_path',
                        type=str,
                        default="corrupt_imgs.txt",
                        required=False,
                        help="""Source dataset path with
                        class imgs inside folders""")
    args = parser.parse_args()
    if args.remove:
        confirm_removal = input(
            "Corrupt files will be removed: Continue (Y/N)?")
        args.remove = True if confirm_removal in {
            'Y', 'y', 'yes', 'Yes'} else False

    if args.remove:
        print("Corrupt files will be removed")
    else:
        print("Corrupt files will NOT be removed")

    validate_imgs(args.source_data_path,
                  args.corrupt_file_list_txt_path,
                  args.remove)


if __name__ == "__main__":
    main()
