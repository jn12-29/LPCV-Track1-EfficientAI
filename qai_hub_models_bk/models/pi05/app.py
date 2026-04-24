# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from lerobot.configs.types import FeatureType
from lerobot.policies.pi05 import PI05Policy

from qai_hub_models.models.protocols import ExecutableModelProtocol


@dataclass
class Pi05AppConfig:
    """
    Lightweight configuration for Pi05App. This mirrors the parts of the
    policy/model config that Pi05App needs, allowing construction without
    passing a full policy object.
    """

    # Number of action steps used by the model.
    # Default value aligns with common PI0 configuration.
    n_action_steps: int = 50

    # Maximum action dimension (e.g., 32 for flow model config).
    max_action_dim: int = 32

    # List of image keys corresponding to visual input features.
    image_keys: list[str] | None = None

    # Actual degree of freedom, not the max action dim (e.g., 7 or 9).
    action_dof: int = 0

    # If True, use the RTC-unrolled expert for inference. When enabled,
    # predict_action_chunk/sample_action require prev_actions.
    use_rtc: bool = False

    @staticmethod
    def from_policy(policy: PI05Policy, use_rtc: bool = False) -> Pi05AppConfig:
        """
        Build Pi05AppConfig from an existing policy. This inspects the
        policy to extract only the fields required by Pi05App.
        """
        cfg_model = policy.model.config
        image_keys = [
            k
            for k, v in cfg_model.input_features.items()
            if v.type == FeatureType.VISUAL and "empty" not in k
        ]
        action_dof = policy.config.output_features["action"].shape[0]
        return Pi05AppConfig(
            n_action_steps=cfg_model.n_action_steps,
            max_action_dim=cfg_model.max_action_dim,
            image_keys=image_keys,
            action_dof=action_dof,
            use_rtc=use_rtc,
        )


class Pi05App(torch.nn.Module):
    """
    Assemble Pi05Collection parts to reproduce the core computation of
    PI05Policy.forward (i.e., FlowMatching forward). This class expects a
    batch coming from LeRobot's dataset and uses the provided policy's
    prepare_* utilities to match PI05Policy.forward preprocessing.

    Expected keys in batch (as observed):
      - "observation.images.*": torch.Tensor [B, 3, H, W]
      - "action" (ACTION): torch.Tensor [B, T, D]
      - "observation.state" (OBS_STATE): torch.Tensor [B, S]
      - "action_is_pad": torch.BoolTensor [B, T] (optional)
      - "task": list[str] (natural language instruction)

    Required forward inputs:
      - noise: torch.Tensor [B, T, Dcfg] produced by the caller
      - time: torch.Tensor [B]

    Forward returns a tuple (loss, loss_dict) where loss is a scalar tensor
    suitable for backward and loss_dict contains "losses_after_forward" with
    shape [B, Tcfg, Dcfg] where Dcfg == model.config.max_action_dim.
    """

    def __init__(
        self,
        config: Pi05AppConfig,
        vit: ExecutableModelProtocol,
        token_emb: ExecutableModelProtocol,
        backbone0_6: ExecutableModelProtocol,
        backbone6_12: ExecutableModelProtocol,
        backbone12_18: ExecutableModelProtocol,
        action_expert: ExecutableModelProtocol,
    ) -> None:
        """
        Initialize Pi05App with model components.

        Parameters
        ----------
        config
            Pi05AppConfig containing model configuration.
        vit
            Vision encoder component.
        token_emb
            Token embedding component.
        backbone0_6
            PaliGemma backbone layers 0-6.
        backbone6_12
            PaliGemma backbone layers 6-12.
        backbone12_18
            PaliGemma backbone layers 12-18.
        action_expert
            Action expert component for denoising.
        """
        super().__init__()

        # Shortcuts to model components.
        self.vit = vit
        self.token_emb = token_emb
        self.backbone0_6 = backbone0_6
        self.backbone6_12 = backbone6_12
        self.backbone12_18 = backbone12_18
        self.action_expert = action_expert

        # Cache a few config bits from the provided flow config. This removes
        # the hard dependency on a policy object while keeping the exact
        # fields Pi05App uses.
        self.n_action_steps: int = config.n_action_steps
        self.max_action_dim: int = config.max_action_dim
        self.image_keys = config.image_keys
        # Actual degree of freedom, not the max action dim (32).
        self.action_dof = config.action_dof
        # Whether to use RTC-unrolled expert during inference.
        self.use_rtc: bool = bool(config.use_rtc)

    def _resize_and_normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize with padding to 224x224 and normalize to [-1, 1].

        Parameters
        ----------
        image
            Tensor [B, C, H, W] with values in [0, 1].

        Returns
        -------
        normalized_image : torch.Tensor
            Tensor [B, C, 224, 224] normalized to [-1, 1].
        """
        if image.ndim != 4:
            raise ValueError(f"[B,C,H,W] expected, got {image.shape}")

        _, _, cur_h, cur_w = image.shape
        tgt_h, tgt_w = 224, 224

        ratio = max(cur_w / tgt_w, cur_h / tgt_h)
        rsz_h = int(cur_h / ratio)
        rsz_w = int(cur_w / ratio)

        image = torch.nn.functional.interpolate(
            image,
            size=(rsz_h, rsz_w),
            mode="bilinear",
            align_corners=False,
        )
        pad_h = max(0, tgt_h - rsz_h)
        pad_w = max(0, tgt_w - rsz_w)

        # Pad (left, right, top, bottom): pad left/top.
        image = torch.nn.functional.pad(image, (pad_w, 0, pad_h, 0), value=0.0)

        # Normalize from [0, 1] to [-1, 1] as expected by SigLIP.
        return image * 2.0 - 1.0

    def populate_prefix(
        self,
        img_ls: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Run vision encoders, token embedding, and PaliGemma backbone chunks
        to build the KV caches and auxiliary tensors required by the
        expert.

        Parameters
        ----------
        img_ls
            List of tensors, each [B, 3, H, W]. Supports a variable
            number of images. The first image is mapped to the primary
            stream; any remaining images are concatenated to the secondary
            stream sequence.
        lang_tokens
            Tensor of token ids [B, Ltok].
        lang_mask
            Bool tensor of attention mask [B, Ltok].

        Returns
        -------
        suffix_sin : torch.Tensor
            Float tensor for RoPE sine embeddings used for
            suffix (actions); shape [B, Lsuffix, Hd].
        suffix_cos : torch.Tensor
            Float tensor for RoPE cosine embeddings used for
            suffix (actions); shape [B, Lsuffix, Hd].
        k_all : list[torch.Tensor]
            list of key-cache tensors for layers 0..17. Each item
            has shape [B, n_heads, Lprefix, head_dim] or per-impl eqv.
        v_all : list[torch.Tensor]
            list of value-cache tensors for layers 0..17. Each item
            has shape [B, n_heads, Lprefix, head_dim] or per-impl eqv.
        full_att_4d : torch.Tensor
            float mask [B, 1, Ls, Lp+Ls] to be used additively
            on attention logits (0 allowed, -1e4 blocked).
        """
        if len(img_ls) == 0:
            raise ValueError("populate_prefix requires at least one image.")

        # Resize to 224x224 with padding and normalize to [-1, 1].
        proc_imgs = [self._resize_and_normalize_image(x) for x in img_ls]

        # Vision encodings (each returns [B, S_img, D]).
        img_embeds = [self.vit(x) for x in proc_imgs]

        # Token embedding packs images + language and produces prefix
        # embeddings/masks and RoPE tensors.
        (
            prefix_emb,
            prefix_att_2d,
            prefix_sin,
            prefix_cos,
            suffix_sin,
            suffix_cos,
            full_att_4d,
        ) = self.token_emb(
            lang_tokens,
            lang_mask.to(dtype=torch.float32),
            *img_embeds,
        )

        # Run PaliGemma backbone to fill per-layer KV caches for the
        # prefix. outX returns (hidden_state, keys..., values...).
        out0 = self.backbone0_6(prefix_emb, prefix_att_2d, prefix_sin, prefix_cos)
        hs0, *rest0 = out0
        n0 = len(rest0) // 2
        k_list0, v_list0 = rest0[:n0], rest0[n0:]

        out1 = self.backbone6_12(hs0, prefix_att_2d, prefix_sin, prefix_cos)
        hs1, *rest1 = out1
        n1 = len(rest1) // 2
        k_list1, v_list1 = rest1[:n1], rest1[n1:]

        rest2 = self.backbone12_18(hs1, prefix_att_2d, prefix_sin, prefix_cos)
        n2 = len(rest2) // 2
        k_list2, v_list2 = rest2[:n2], rest2[n2:]

        # Concatenate caches across all 18 layers.
        k_all = list(k_list0) + list(k_list1) + list(k_list2)
        v_all = list(v_list0) + list(v_list1) + list(v_list2)

        return suffix_sin, suffix_cos, k_all, v_all, full_att_4d

    def denoise_step(
        self,
        k_all: list[torch.Tensor],
        v_all: list[torch.Tensor],
        suffix_sin: torch.Tensor,
        suffix_cos: torch.Tensor,
        x_t: torch.Tensor,
        time_step: torch.Tensor,
        full_att_4d: torch.Tensor,
        prev_chunk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply one denoising step using the cached prefix and the action
        expert, mirroring PI0FlowMatching.denoise_step behavior.

        Parameters
        ----------
        k_all
            List of key-cache tensors for layers 0..17. Each item
            typically has shape [B, n_heads, Lprefix, head_dim].
        v_all
            List of value-cache tensors for layers 0..17. Each item
            typically has shape [B, n_heads, Lprefix, head_dim].
        suffix_sin
            Float tensor RoPE sine embedding for suffix with
            shape [B, Lsuffix, Hd].
        suffix_cos
            Float tensor RoPE cosine embedding for suffix with
            shape [B, Lsuffix, Hd].
        x_t
            Float tensor of noisy actions [B, Tcfg, Dcfg].
        time_step
            Float tensor [B] with values in [0, 1].
        full_att_4d
            Float additive mask [B,1,Ls,Lp+Ls] for attention.
        prev_chunk
            When use_rtc is True, the previous action chunk
            with shape [B, Tcfg, Dcfg]. Ignored otherwise.

        Returns
        -------
        updated_actions : torch.Tensor
            Tensor [B, Tcfg, Dcfg] of updated actions x_{t+dt}.
        """
        action_kwargs: dict[str, torch.Tensor] = {
            "rope_emb_sin": suffix_sin,
            "rope_emb_cos": suffix_cos,
            "x_t": x_t.to(torch.float32),
            "full_att_4d": full_att_4d,
            "time_step": time_step,
        }
        for i, k in enumerate(k_all):
            action_kwargs[f"key_cache_l{i}"] = k.to(torch.float32)
        for i, v in enumerate(v_all):
            action_kwargs[f"value_cache_l{i}"] = v.to(torch.float32)

        if self.use_rtc:
            if prev_chunk is None:
                raise ValueError("prev_chunk must be provided when use_rtc is True.")
            return self.action_expert(
                prev_chunk=prev_chunk.to(torch.float32),
                **action_kwargs,
            )

        # Expert returns x_{t+dt} after an internal Euler step.
        return self.action_expert(**action_kwargs)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor | Any],
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        prev_actions: torch.Tensor | None = None,
        truncate_action_by_dof: bool = True,
    ) -> torch.Tensor:
        lang_tokens = batch["observation.language.tokens"]  # [B, 200]
        assert self.image_keys is not None
        img_ls = [batch[key] for key in self.image_keys]  # [B, 3, 224, 224]
        # [B, 200] (float)
        lang_mask = batch["observation.language.attention_mask"].to(torch.float32)

        if self.use_rtc and prev_actions is None:
            raise ValueError("prev_actions is required when use_rtc is True.")

        # Sample actions using the model (no robot state needed in Pi05).
        actions = self.sample_action(
            img_ls=img_ls,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
            noise=noise,
            num_steps=num_steps,
            prev_actions=prev_actions,
        )

        # Truncate to the policy-configured output action dim.
        if not truncate_action_by_dof:
            return actions
        return actions[:, :, : self.action_dof]

    @torch.no_grad()
    def sample_action(
        self,
        img_ls: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
        prev_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform inference by running prefix population once and then
        applying Euler updates for a fixed number of steps, similar to
        PI0FlowMatching.sample_actions.

        Parameters
        ----------
        img_ls
            List of [B, 3, H, W] images.
        lang_tokens
            Tensor [B, Ltok] language tokens.
        lang_mask
            Bool tensor [B, Ltok] language mask.
        noise
            Optional float tensor [B, Tcfg, Dcfg]. If None, a
            standard normal tensor is sampled.
        num_steps
            Integer Euler steps. Default is 10.
        prev_actions
            When use_rtc is True, provide the previous
            chunk [B, Tcfg, Dcfg] whose first entry corresponds to the
            next action to execute.

        Returns
        -------
        denoised_actions : torch.Tensor
            Float tensor [B, Tcfg, Dcfg] of denoised actions that are
            already denormalized back to the action space.
        """
        if len(img_ls) == 0:
            raise ValueError("sample_action requires at least one image.")

        if self.use_rtc and prev_actions is None:
            raise ValueError("prev_actions is required when use_rtc is True.")

        device = img_ls[0].device
        bsize = lang_tokens.shape[0]

        (
            suffix_sin,
            suffix_cos,
            k_all,
            v_all,
            full_att_4d,
        ) = self.populate_prefix(
            img_ls=img_ls,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask.to(torch.float32),
        )

        # Initialize noise if not provided.
        if noise is None:
            actions_shape = (
                bsize,
                self.n_action_steps,
                self.max_action_dim,
            )
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=actions_shape,
                dtype=torch.float32,
                device=device,
            )

        # Euler integration from t=1 down to ~0.
        dt = torch.tensor(
            -1.0 / float(num_steps),
            dtype=torch.float32,
            device=device,
        )
        x_t = noise
        t_cur = torch.tensor(1.0, dtype=torch.float32, device=device)
        while t_cur >= -dt / 2:
            time_b = t_cur.expand(bsize)  # [B]
            # Expert returns x_{t+dt}; advance t here only.
            x_t = self.denoise_step(
                k_all=k_all,
                v_all=v_all,
                suffix_sin=suffix_sin,
                suffix_cos=suffix_cos,
                x_t=x_t,
                time_step=time_b,
                full_att_4d=full_att_4d,
                prev_chunk=prev_actions,
            )
            t_cur = t_cur + dt

        return x_t
