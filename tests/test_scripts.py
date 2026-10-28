"""Tests for the data-prep scripts in scripts/"""

from scripts.duplicate_data import duplicate_data


def test_duplicate_data_reaches_the_target_number(tmp_path, create_and_save_dummy_imgs):
    """Each class gets exactly target_number images, also when the target dir is the raw dir"""
    create_and_save_dummy_imgs(str(tmp_path / "raw"), n_cls=2, n_imgs_p_cls=3)
    duplicate_data(str(tmp_path / "raw"), str(tmp_path / "copies"), target_number=7)
    duplicate_data(str(tmp_path / "raw"), str(tmp_path / "raw"), target_number=7)
    for target in ("copies", "raw"):
        assert [len(list(class_dir.iterdir())) for class_dir in sorted((tmp_path / target).iterdir())] == [7, 7]
