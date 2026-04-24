# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
import re
import sys
from argparse import RawTextHelpFormatter

from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.utils import aws, printing
from qai_hub_models.utils.path_helpers import MODEL_IDS


def _get_int_asset_version(asset_version: str) -> str | None:
    try:
        return f"v{int(asset_version)}"
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="""
Print or upload asset storage for a given model or dataset.

Examples:

# Prints the files in asset version 1 for the mobilenet_v2 model
python upload_asset_store.py --name mobilenet_v2 --asset-version 1

# Uploads my_file.txt to asset version 1 for the mobilenet_v2 model
python upload_asset_store.py --name mobilenet_v2 --asset-version 1 --upload-files ../my_file.txt

# Uploads my_folder and all contents to asset version 1 for the COCO dataset.
python upload_asset_store.py --name coco --type dataset --asset-version 1 --upload-files /path/to/my_folder

# Uploads my_folder and all contents to asset version qnn_2.28.2-v79-soc69 for the mistral_3b_quantized model.
# This data is stored in a private bucket, and is not publicly accessible.
python upload_asset_store.py --private --name mistral_3b_quantized --asset-version qnn_2.28.2-v79-soc69 --upload-files /path/to/my_folder

""",
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["model", "dataset"],
        default="model",
        help="Type of object to upload assets for.\n\n",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name of object (model or dataset) to upload assets for.\n\n",
    )
    parser.add_argument(
        "--asset-version",
        "-v",
        type=str,
        help="""Version of the model/dataset assets to modify. Can be:
Public Assets: (default upload location)
  - An integer
        For generic assets. This is usually the correct choice.

  - "web-assets", "w", "web"
        Only applicable for models.
        Used for WEBSITE PHOTOS. NEVER USE FOR GENERIC model / dataset FILES.

Private Assets: (when using --private)
  - An integer
        For generic assets.

  -  "qnn_<qnn_version>-v<hexagon_version>-soc<soc_version>"
        Only applicable for models.
        For context binary / genie config / SoC config distributions of proprietary models.

""",
    )
    parser.add_argument(
        "--private",
        "-p",
        action="store_true",
        help=f"If set, modifies a private storage bucket ({aws.QAIHM_PRIVATE_S3_BUCKET}) instead of the publicly accessible bucket.\n\n",
    )
    parser.add_argument(
        "--upload-files",
        "-u",
        type=str,
        nargs="+",
        help="""Model folders / files to upload to the asset store.
Uploaded files & folders are referenced by their name when getting files in the asset store (CachedWebModelAsset.from_asset_store).

For example:
* if you upload /path/to/my/file.txt, the resulting file can be referenced as:
    - `file.txt`

* if you upload /path/to/my/folder, files within the folder can be referenced as:
    - `folder/my_file.txt`
    - `folder/subfolder/my_file2.txt`
    - etc.

If a file already exists at a location, an error will be thrown.
If a folder already exists at a location to which a folder is being uploaded, the existing folder will be merged with the new folder.
""",
    )
    args = parser.parse_args()

    # Parse inputs
    name: str = args.name
    asset_type: str = args.type
    is_model = asset_type == "model"
    is_dataset = asset_type == "dataset"
    asset_version: str = args.asset_version
    private: bool = args.private
    upload_files: list[str] = args.upload_files

    # Validate args
    is_numeric_asset_version = False
    if private:
        if x := _get_int_asset_version(asset_version):
            asset_version = x
            is_numeric_asset_version = True
        elif not re.match(r"qnn_\d+\.\d+(\.\d+)?-v\d+-soc\d+", asset_version):
            raise ValueError(
                "Asset version for unpublished models must either be an integer or match the following format: 'qnn_<qnn_version>-v<hexagon_version>-soc<soc_version>'"
            )

    elif asset_version in ["w", "web", "web-assets"]:
        asset_version = "web-assets"

    elif x := _get_int_asset_version(asset_version):
        asset_version = x
        is_numeric_asset_version = True
    else:
        raise ValueError(
            "Public asset version must either be 'web-assets (web | w)' or an integer."
        )

    if is_dataset and not is_numeric_asset_version:
        raise ValueError("Datasets only support integer asset versions.")

    if is_model and name not in MODEL_IDS:
        raise ValueError(f"{name} is not a valid model ID.")
    if is_dataset and name not in DATASET_NAME_MAP:
        raise ValueError(f"{name} is not a valid dataset name.")

    # Public / Private bucket and s3 root
    bucket_name = aws.QAIHM_PRIVATE_S3_BUCKET if private else aws.QAIHM_PUBLIC_S3_BUCKET
    s3_root = os.path.join("qai-hub-models", f"{asset_type}s", name, asset_version)

    # Get boto3 objs
    bucket, is_admin = aws.get_qaihm_s3_or_exit(bucket_name)

    # Get file changes
    (
        unmodified_s3_objs,
        files_to_upload,
        _,
        _,
    ) = aws.get_files_to_upload_remove(bucket, s3_root, upload_files)

    # Print proposed changes
    print()
    print(f"S3 Bucket: {bucket.name}")
    printing.print_file_tree_changes(
        s3_root,
        files_unmodified=[asset.key for asset in unmodified_s3_objs],
        files_added=[x[1] for x in files_to_upload],
    )
    print()

    # Verify user wants changes
    if files_to_upload and not is_admin:
        print(
            "WARNING: Once uploaded, files cannot be replaced or deleted without the help of a project maintainer. TREAD CAREFULLY!"
        )
    if files_to_upload and not private:
        print(
            "\n!!!!!! All uploaded files will be immediately accessible by the public. DO NOT UPLOAD CCI !!!!!!\n"
        )

    if files_to_upload:
        response = input("Execute the above diff? (y): ")
        if response.lower() not in ["y", "yes"]:
            sys.exit(0)
        print()

    # Execute changes
    for local_file_path, asset_key in files_to_upload:
        aws.s3_multipart_upload(bucket, asset_key, local_file_path, not private)


if __name__ == "__main__":
    main()
