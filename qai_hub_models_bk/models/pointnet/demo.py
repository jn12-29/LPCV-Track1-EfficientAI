# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.pointnet import App
from qai_hub_models.models.pointnet.model import MODEL_ASSET_VERSION, MODEL_ID, Pointnet
from qai_hub_models.utils.args import get_model_cli_parser, get_on_device_demo_parser
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

DATASET_ADDR = str(
    CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "dataset/chair/test/chair_0890.off"
    ).fetch()
).split("chair")[0]


def pointnet_demo(
    model_type: type[Pointnet],
    default_input: str,
    is_test: bool = False,
) -> None:
    """
    Runs a demo for PointNet model to classify a 3D point cloud.

    This function:
    - Loads a pre-trained PointNet model.
    - Loads and transforms point cloud data from the given path.
    - Performs inference to predict the class of the object in the point cloud.
    - Prints the predicted class if not in test mode.

    Parameters
    ----------
    model_type
        The PointNet model class to use.
    default_input
        Path to the folder containing point cloud data.
    is_test
        If True, runs in test mode without printing output.
    """
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_input,
        help="Path to the Point Cloud Directory",
    )
    args = parser.parse_args([] if is_test else None)
    model = model_type.from_pretrained()
    app = App(model=model)
    test_loader = app.load_cloud_data(args.dataset)
    predicted = app.predict(test_loader=test_loader)
    if not is_test:
        print("output: ", predicted)


def main(is_test: bool = False) -> None:
    pointnet_demo(Pointnet, DATASET_ADDR, is_test)


if __name__ == "__main__":
    main()
