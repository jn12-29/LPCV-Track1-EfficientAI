# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide

from qai_hub_models.models._shared.sam.app import SAMApp, SAMInputImageLayout
from qai_hub_models.models._shared.sam.utils import show_image
from qai_hub_models.models.sam.model import (
    BASE_MODEL_TYPE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    SAM,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.evaluate import EvalMode

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "truck.jpg"
)


# Run SAM end-to-end model on given image.
# The demo will output image with segmentation mask applied for input points
def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(SAM)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )
    parser.add_argument(
        "--point-coordinates",
        type=str,
        default="1342,1011;",
        help="Comma separated x and y coordinate. Multiple coordinate separated by `;`."
        " e.g. `x1,y1;x2,y2`. Default: `500,375;`",
    )
    get_on_device_demo_parser(parser, add_output_dir=True)

    args = parser.parse_args(["--model-type", BASE_MODEL_TYPE] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    coordinates: list[str] = list(filter(None, args.point_coordinates.split(";")))

    # Load Application
    wrapper = SAM.from_pretrained(args.model_type)

    if args.eval_mode == EvalMode.ON_DEVICE:
        encoder_splits_and_decoder = demo_model_components_from_cli_args(
            SAM, MODEL_ID, args
        )
        sam_encoder_splits = list(encoder_splits_and_decoder[:-1])
        sam_decoder = encoder_splits_and_decoder[-1]
    else:
        sam_encoder_splits = list(wrapper.encoder_splits)
        sam_decoder = wrapper.decoder

    app = SAMApp(
        wrapper.sam.image_encoder.img_size,
        wrapper.sam.mask_threshold,
        SAMInputImageLayout[wrapper.sam.image_format],
        sam_encoder_splits,  # type: ignore[arg-type]
        sam_decoder,  # type: ignore[arg-type]
        ResizeLongestSide,
        wrapper.sam.pixel_mean,
    )

    # Load Image
    image = load_image(args.image)

    # Point segmentation using decoder
    print("\n** Performing point segmentation **\n")

    # Input points
    input_coords = []
    input_labels = []

    for coord in coordinates:
        coord_split = coord.split(",")
        if len(coord_split) != 2:
            raise RuntimeError(
                f"Expecting comma separated x and y coordinate. Provided {coord_split}."
            )

        input_coords.append([int(coord_split[0]), int(coord_split[1])])
        # Set label to `1` to include current point for segmentation
        input_labels.append(1)

    # Generate masks with given input points
    generated_mask, *_ = app.predict_mask_from_points(
        image, torch.Tensor(input_coords), torch.Tensor(input_labels)
    )

    if not is_test:
        show_image(image, np.asarray(generated_mask), input_coords, args.output_dir)


if __name__ == "__main__":
    main()
