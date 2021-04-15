import os
import glob
import imageio
import argparse
import numpy as np
from tqdm import tqdm
from util import _fix_path_for_globbing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
                        '--raw_img_path',
                        type=str,
                        required=True,
                        help="""Source dataset path with
                        class imgs inside folders""")
    parser.add_argument('-t',
                        '--target_npz_img_path',
                        type=str,
                        required=True,
                        help="""Target directory path to save
                        npz files in""")
    args = parser.parse_args()
    generate_npz_files(args.raw_img_path, args.target_npz_img_path)


def generate_npz_files(raw_img_path, npz_img_path) -> None:
    os.makedirs(npz_img_path, exist_ok=True)
    dir_list = glob.glob(_fix_path_for_globbing(raw_img_path))

    for i in tqdm(range(len(dir_list))):
        npz_name = 1
        dir_path = dir_list[i]
        img_list = glob.glob(dir_path + "/*")

        class_name = dir_path.split("/")[-1]
        class_path = os.path.join(npz_img_path, class_name)
        os.makedirs(class_path, exist_ok=True)
        print(f"Converting class {class_name} to npz")
        for j in tqdm(range(len(img_list))):
            img_name = img_list[j]
            try:
                img = imageio.imread(img_name, pilmode="RGB")
                img = img[..., :3]
                img_npz_path = os.path.join(class_path, str(npz_name).zfill(6))
                np.savez(img_npz_path, image=img)
                npz_name += 1
            except Exception as e:
                print(f"{e}. imageio could not read file {img_name}")


if __name__ == "__main__":
    main()
