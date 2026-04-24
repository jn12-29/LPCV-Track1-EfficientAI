# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
from typing import TypeVar, cast

import torch
from segment_anything import sam_model_registry
from segment_anything.modeling import image_encoder as sam_image_encoder
from segment_anything.modeling.image_encoder import Block as SAM_Encoder_Block
from segment_anything.modeling.mask_decoder import MLP as SAMMaskDecoderMLP
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.transformer import (
    TwoWayAttentionBlock,
    TwoWayTransformer,
)
from segment_anything.utils.onnx import SamOnnxModel
from typing_extensions import Self

from qai_hub_models.models._shared.sam.model_patches import (
    Conv2DInplaceLinearSAMMaskDecoderMLP,
    Conv2DInplaceLinearSAMTransformerMLPBlock,
    SplitHeadSAMDecoderAttention,
    sam_decoder_predict_masks,
)
from qai_hub_models.models.sam.model_patches import (
    SplitHeadSAMEncoderAttention,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_model import (
    BaseModel,
    CollectionModel,
    PretrainedCollectionModel,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)
from qai_hub_models.utils.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2

BASE_MODEL_TYPE = "vit_b"
LARGE_MODEL_TYPE = "vit_l"
HUGE_MODEL_TYPE = "vit_h"
DEFAULT_MODEL_TYPE = BASE_MODEL_TYPE

MODEL_REGISTERY = {
    BASE_MODEL_TYPE: "sam_vit_b_01ec64.pth",  # 91M params
    LARGE_MODEL_TYPE: "sam_vit_l_0b3195.pth",  # 308M params
    HUGE_MODEL_TYPE: "sam_vit_h_4b8939.pth",  # 636M params
}

# Patch Encoder to use 5D Window Partition (rather than 6D)
sam_image_encoder.window_partition = window_partition_5d
sam_image_encoder.window_unpartition = window_unpartition_5d


class SAMEncoderPart(BaseModel):
    """Exportable SAM encoder that can be split into several parts."""

    def __init__(
        self,
        sam: Sam,
        include_embedding: bool = True,
        include_transformer_blocks: tuple[int, int | None] | None = (0, None),
        include_neck: bool = True,
    ) -> None:
        super().__init__()
        self.sam = sam
        self.include_embedding = include_embedding
        self.include_neck = include_neck
        self.include_transformer_blocks = include_transformer_blocks

        if (
            not include_embedding
            and not include_transformer_blocks
            and not include_neck
        ):
            raise ValueError("You must include at least 1 slice of the encoder")

    def forward(self, xOrImage: torch.Tensor) -> torch.Tensor:
        """
        Run SAM Image encoder and returns image embeddings

        Parameters
        ----------
        xOrImage
            If self.include_embedding is true:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]

            Otherwise:
                An intermediate tensor that is an output of a different "slice" of the encoder.
                Shape (batch_size, 64, 64, 768)

        Returns
        -------
        torch.Tensor
            If self.include_neck is True, returns:
            image_embeddings
                Shape (1, 256, 64, 64).

            If self.include_neck is False, returns:
            output
                An intermediate tensor output of this "slice" of the encoder.
                Shape (batch_size, 64, 64, 768).
        """
        x = xOrImage
        if self.include_embedding:
            x = (x - self.sam.pixel_mean) * (1 / self.sam.pixel_std)
            x = self.sam.image_encoder.patch_embed(x)
            if self.sam.image_encoder.pos_embed is not None:
                x = x + self.sam.image_encoder.pos_embed

        if self.include_transformer_blocks is not None:
            for blk in self.sam.image_encoder.blocks[
                self.include_transformer_blocks[0] : self.include_transformer_blocks[1]
            ]:
                x = blk(x)

        if self.include_neck:
            x = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2))

        return x

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        encoder_img_height: int = 1024,  # self.sam.image_encoder.img_size
        encoder_img_width: int = 1024,  # self.sam.image_encoder.img_size
        include_embedding: bool = True,
        embedding_size: int = 768,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        if include_embedding:
            return {
                "image": TensorSpec(
                    shape=(batch_size, 3, encoder_img_height, encoder_img_width),
                    dtype="float32",
                    io_type=IoType.IMAGE,
                    image_metadata=ImageMetadata(
                        color_format=ColorFormat.RGB,
                        value_range=(0.0, 1.0),
                    ),
                ),
            }
        return {
            "x": TensorSpec(
                shape=(batch_size, 64, 64, embedding_size),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def _get_input_spec_for_instance(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        """Override for model.get_input_spec() when called on instances of this class."""
        return self.__class__.get_input_spec(
            batch_size,
            self.sam.image_encoder.img_size,
            self.sam.image_encoder.img_size,
            self.include_embedding,
            self.sam.image_encoder.blocks[0].attn.in_feature,
        )

    @staticmethod
    def get_channel_last_inputs(include_embedding: bool = True) -> list[str]:
        if include_embedding:
            return list(
                SAMEncoderPart.get_input_spec(
                    include_embedding=include_embedding
                ).keys()
            )
        return []

    def _get_channel_last_inputs_for_instance(self) -> list[str]:
        return self.__class__.get_channel_last_inputs(self.include_embedding)

    @staticmethod
    def get_channel_last_outputs(include_neck: bool = True) -> list[str]:
        # Output of encoder parts should not be channel last.
        # This actually inserts a transpose when one is not needed.
        return ["image_embeddings"] if include_neck else []

    def _get_channel_last_outputs_for_instance(self) -> list[str]:
        return self.__class__.get_channel_last_outputs(self.include_neck)

    @staticmethod
    def get_output_names(include_neck: bool = True) -> list[str]:
        return (
            ["image_embeddings"]
            if include_neck
            else ["intermediate_SAM_encoder_output"]
        )

    def _get_output_names_for_instance(self) -> list[str]:
        return SAMEncoderPart.get_output_names(self.include_neck)

    @classmethod
    def from_pretrained(cls, model_type: str = BASE_MODEL_TYPE) -> Self:
        return SAMLoader.load(
            encoder_cls=cls,
            model_type=model_type,
            single_mask_mode=True,
            num_encoder_splits=0,
        )[1][0]


class SAMDecoder(BaseModel):
    """
    Adapted from from segment_anything.utils.onnx.SamOnnxModel with modifications.

    This removes output mask resizing. Because this requires a dynamic shape to accomplish
    in the network, it's better to do this as a postprocessing step rather than in the inference
    framework itself.
    """

    def __init__(self, sam: Sam, return_single_mask: bool) -> None:
        super().__init__(sam)
        self.model: Sam
        self.embed_size = self.model.prompt_encoder.image_embedding_size
        self.img_size = self.model.image_encoder.img_size
        self.return_single_mask = return_single_mask

    def _embed_masks(self, input_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Lifted from segment_anything.utils.onnx.SamOnnxModel

        Modified to remove ops based on whether input_mask is set.
        """
        if input_mask is not None:
            return self.model.prompt_encoder.mask_downscaling(input_mask)
        return torch.zeros(
            1, 1, *self.embed_size
        ) + self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SAM lightweight decoder and return generated mask for given points

        Parameters
        ----------
        image_embeddings
            Image embeddings generated by Encoder. Shape [1, emb_dim, emb_size, emb_size]
        point_coords
            Point coordinates from input image for segmentation. Shape [1, k, 2]
        point_labels
            Point Labels to select/de-select given point for segmentation. Shape [1, k]
            e.g. Corresponding value is 1 if this point is to be included, otherwise 0
        mask_input
            Input mask to consider for segmentation. Shape [1, 1, 4 * self.embed_size, 4 * self.embed_size]
            If using point based segmentation, this is unused.

        Returns
        -------
        masks : torch.Tensor
            Torch.Tensor of shape [1, k, 256, 256] where k = number of points.
        scores : torch.Tensor
            Torch.Tensor of shape [1, k] where k = number of points.
        """
        sparse_embedding = SamOnnxModel._embed_points(self, point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input)

        masks, scores = sam_decoder_predict_masks(
            self.model.mask_decoder,
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.return_single_mask:
            masks, scores = SamOnnxModel.select_masks(
                self, masks, scores, point_coords.shape[1]
            )

        return masks, scores

    def _get_input_spec_for_instance(
        self: SAMDecoder,
        has_mask_input: bool = False,
        num_of_points: int = 2,
    ) -> InputSpec:
        """
        Override for model.get_input_spec() when called on instances of this class.

        The initializer for BaseModel will automatically override get_input_spec
        with this function when the class is instantiated.
        """
        return self.__class__.get_input_spec(
            has_mask_input,
            num_of_points,
            self.model.prompt_encoder.embed_dim,
            self.embed_size[0],
            self.embed_size[1],
        )

    @staticmethod
    def get_input_spec(
        has_mask_input: bool = False,
        num_of_points: int = 2,
        embed_dim: int = 256,
        image_embedding_height: int = 64,
        image_embedding_width: int = 64,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        embed_size = (image_embedding_height, image_embedding_width)
        mask_input_size = tuple([4 * x for x in embed_size])

        input_spec: InputSpec = {
            "image_embeddings": TensorSpec(
                shape=(1, embed_dim, *embed_size),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "point_coords": TensorSpec(
                shape=(1, num_of_points, 2),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
            "point_labels": TensorSpec(
                shape=(1, num_of_points),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }
        if has_mask_input:
            input_spec["mask_input"] = TensorSpec(
                shape=(1, 1, *mask_input_size),
                dtype="float32",
                io_type=IoType.TENSOR,
            )
            input_spec["has_mask_input"] = TensorSpec(
                shape=(1,),
                dtype="float32",
                io_type=IoType.TENSOR,
            )
        return input_spec

    @staticmethod
    def get_channel_last_inputs(has_mask_input: bool = False) -> list[str]:
        out = ["image_embeddings"]
        if has_mask_input:
            out.append("mask_input")
        return out

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["masks"]

    @staticmethod
    def get_output_names() -> list[str]:
        return ["masks", "scores"]

    @classmethod
    def from_pretrained(cls, model_type: str = BASE_MODEL_TYPE) -> Self:
        return SAMLoader.load(
            decoder_cls=cls,
            model_type=model_type,
            single_mask_mode=True,
            num_encoder_splits=0,
        )[2]


EncoderPartT = TypeVar("EncoderPartT", bound=SAMEncoderPart)
DecoderT = TypeVar("DecoderT", bound=SAMDecoder)


class SAMLoader:
    """Helper class for loading and preparing a HTP-compatible SAM model."""

    @staticmethod
    def load(
        model_type: str = BASE_MODEL_TYPE,
        single_mask_mode: bool = True,
        num_encoder_splits: int = 0,
        encoder_cls: type[EncoderPartT] = SAMEncoderPart,  # type: ignore[assignment]
        decoder_cls: type[DecoderT] = SAMDecoder,  # type: ignore[assignment]
    ) -> tuple[Sam, list[EncoderPartT], DecoderT]:
        # https://github.com/qcom-ai-hub/tetracode/issues/14357
        # sam test.py::test_e2e_numerical fails if using num_splits=0 for encoder
        # (vit-b small variant)

        # Even 10 part encoder splitting fails at the last split
        # https://dev.aihub.qualcomm.com/jobs/jpxkrm6l5
        # (vit-b huge variant)
        sam = SAMLoader._load_sam_from_repo(model_type)
        SAMLoader._patch_sam_for_qnn_comatibility(sam)
        encoder_splits = SAMLoader._split_sam_encoder(
            sam, num_encoder_splits, encoder_cls
        )
        decoder = decoder_cls(sam, single_mask_mode)
        return sam, encoder_splits, decoder

    @staticmethod
    def _load_sam_from_repo(model_type: str = DEFAULT_MODEL_TYPE) -> Sam:
        """
        Get the SAM described by the given model type.
        SAM will be patched for QNN compatibility.
        """
        if model_type not in MODEL_REGISTERY:
            raise RuntimeError(f"Weights not found for model type `{model_type}`.")

        asset = CachedWebModelAsset(
            f"https://dl.fbaipublicfiles.com/segment_anything/{MODEL_REGISTERY[model_type]}",
            MODEL_ID,
            MODEL_ASSET_VERSION,
            f"{MODEL_REGISTERY[model_type]}",
        )
        asset.fetch()
        return sam_model_registry[model_type](asset.path)

    @staticmethod
    def _patch_sam_for_qnn_comatibility(sam: Sam, patch_encoder: bool = True) -> None:
        """Apply a patch to the SAM class for compatibility with QNN."""
        # Normalize pixel_mean and pixel_std for fp ([0, 1]) input
        # Allows network inputs to be float instead of int.
        sam.pixel_mean = sam.pixel_mean / 255.0  # [0-255] -> [0, 1]
        sam.pixel_std = sam.pixel_std / 255.0  # [0-255] -> [0, 1]

        ###
        # Patch the graph for compatibility with QNN.
        #
        # All below optimizations either optimize for QNN inference speed,
        # or fix failures that occur when compiling to QNN.
        ###
        if patch_encoder:
            for block in sam.image_encoder.blocks:
                assert isinstance(block, SAM_Encoder_Block)
                block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)
                block.attn = SplitHeadSAMEncoderAttention(block.attn)

        sam.mask_decoder.predict_masks = functools.partial(
            sam_decoder_predict_masks, sam.mask_decoder
        )
        for i in range(len(sam.mask_decoder.output_hypernetworks_mlps)):
            mlp = cast(SAMMaskDecoderMLP, sam.mask_decoder.output_hypernetworks_mlps[i])
            sam.mask_decoder.output_hypernetworks_mlps[i] = (
                Conv2DInplaceLinearSAMMaskDecoderMLP(mlp)
            )
        sam.mask_decoder.iou_prediction_head = Conv2DInplaceLinearSAMMaskDecoderMLP(
            sam.mask_decoder.iou_prediction_head
        )

        transformer = cast(TwoWayTransformer, sam.mask_decoder.transformer)
        transformer.final_attn_token_to_image = SplitHeadSAMDecoderAttention(
            transformer.final_attn_token_to_image
        )
        for block in transformer.layers:
            block = cast(TwoWayAttentionBlock, block)
            block.self_attn = SplitHeadSAMDecoderAttention(block.self_attn)
            block.cross_attn_token_to_image = SplitHeadSAMDecoderAttention(
                block.cross_attn_token_to_image
            )
            block.cross_attn_image_to_token = SplitHeadSAMDecoderAttention(
                block.cross_attn_image_to_token
            )
            block.mlp = Conv2DInplaceLinearSAMTransformerMLPBlock(block.mlp)

    @staticmethod
    def _split_sam_encoder(
        sam: Sam,
        num_splits: int,
        encoder_cls: type[EncoderPartT] = SAMEncoderPart,  # type: ignore[assignment]
    ) -> list[EncoderPartT]:
        """
        Split the given SAM model's encoder into smaller pieces (components).
        This is done for HTP compatibility.

        The model is too big to fit on the HTP as one graph,
        but each component can fit on the HTP individually.

        Discussion:
            Each encoder part is assigned an equal portion of N
            sequential attention blocks, resulting in num_splits + 1 instances of SAMEncoderPart.

            Returns num_splits encoders, each with the same number of attention blocks.
            The encoders will be returned in the order they should be executed.

            The first encoder will contain the image embedding step, and the last encoder will contain the encoder "neck".

        Parameters
        ----------
        sam
            The SAM model instance.
        num_splits
            Number of times the encoder should be split. Must be 0 or higher.
        encoder_cls
            Encoder Class type to instantiate for each split.

        Returns
        -------
        encoder_splits : list[EncoderPartT]
            List of SAMEncoderPart instances in execution order.
        """
        encoder_splits: list[EncoderPartT] = []
        if num_splits == 0:
            # Single end-to-end encoder (default constructor)
            encoder_splits.append(encoder_cls(sam))
        else:
            # Split the encoder into several models.
            # Each model will have a portion of the transformer blocks.

            # Get number of transformer blocks that should be included in each model
            n_transformer_blocks = len(sam.image_encoder.blocks)
            block_split_len = n_transformer_blocks // (num_splits + 1)

            # Generate split indices
            split_idx: list[tuple[int, int]] = [
                (i * block_split_len, (i + 1) * block_split_len)
                for i in range(num_splits + 1)
            ]

            # Add make sure final split if self.num_encoder_splits + 1 is the last transformer block.
            # This is necessary if self.num_encoder_splits + 1 does not evenly divide n_transformer_blocks
            split_idx[-1] = (split_idx[-1][0], n_transformer_blocks)

            # Add first encoder. Includes embedding + transformer blocks.
            encoder_splits.append(
                encoder_cls(
                    sam,
                    include_embedding=True,
                    include_transformer_blocks=split_idx[0],
                    include_neck=False,
                )
            )

            # Add several encoders consisting of only transformer blocks..
            encoder_splits.extend(
                encoder_cls(
                    sam,
                    include_embedding=False,
                    include_transformer_blocks=split_idx[i],
                    include_neck=False,
                )
                for i in range(1, len(split_idx) - 1)
            )

            # Add final encoder. Includes transformer blocks + neck.
            encoder_splits.append(
                encoder_cls(
                    sam,
                    include_embedding=False,
                    include_transformer_blocks=split_idx[-1],
                    include_neck=True,
                )
            )

        return encoder_splits


class SAM(PretrainedCollectionModel):
    def __init__(
        self, sam: Sam, encoder_splits: list[SAMEncoderPart], decoder: SAMDecoder
    ) -> None:
        super().__init__(*[*encoder_splits, decoder])
        self.sam = sam
        self.encoder_splits = encoder_splits
        self.decoder = decoder

    @staticmethod
    def from_pretrained(
        model_type: str = DEFAULT_MODEL_TYPE, single_mask_mode: bool = True
    ) -> SAM:
        cls: type[SAM]
        if model_type == BASE_MODEL_TYPE:
            cls = _SAMBase
        elif model_type == LARGE_MODEL_TYPE:
            cls = _SAMLarge
        elif model_type == HUGE_MODEL_TYPE:
            cls = _SAMHuge
        else:
            raise NotImplementedError(f"Unknown SAM model type {model_type}")
        return cls(
            *SAMLoader.load(
                model_type,
                single_mask_mode,
                num_encoder_splits=len(cls.component_class_names) - 2,
            )
        )


@CollectionModel.add_component(SAMEncoderPart, "sam_encoder")
@CollectionModel.add_component(SAMDecoder, "sam_decoder")
class _SAMBase(SAM):
    pass


@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_1")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_2")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_3")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_4")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_5")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_6")
@CollectionModel.add_component(SAMDecoder, "sam_decoder")
class _SAMLarge(SAM):
    pass


@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_1")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_2")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_3")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_4")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_5")
@CollectionModel.add_component(SAMEncoderPart, "sam_encoder_part_6")
@CollectionModel.add_component(SAMDecoder, "sam_decoder")
class _SAMHuge(SAM):
    pass
