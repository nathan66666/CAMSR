from os import path as osp

from basicsr.utils import scandir


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, "r") as fin:
        gt_names = [line.strip().split(" ")[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f"{filename_tmpl.format(basename)}{ext}"
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [input_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [input_key, gt_key]. But got {len(keys)}"
    )
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f"{input_key} and {gt_key} datasets have different number of images: "
        f"{len(input_paths)}, {len(gt_paths)}."
    )
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))

        input_name = f"{filename_tmpl.format(basename)}{ext}"

        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f"{input_name} is not in {input_key}_paths."
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f"{input_key}_path", input_path), (f"{gt_key}_path", gt_path)])
        )
    return paths
