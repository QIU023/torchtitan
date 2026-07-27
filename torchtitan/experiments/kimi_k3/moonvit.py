# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT-V2: Kimi K3's vision tower (report sec 2.4).

Every structural choice below is pinned to the released ``vision_config`` or to
the report's architecture paragraph, both quoted at the field that encodes them:

    vt_num_hidden_layers   27      vt_hidden_size      1024
    vt_num_attention_heads 12      qkv_hidden_size     1536
    vt_intermediate_size   4096    patch_size          14
    norm_type              rmsnorm activation_func     gelu_pytorch_tanh
    attn_bias / linear_bias / patch_embed_proj_bias    all False
    pos_emb_type    divided_fixed  init_pos_emb_{time,height,width}  4, 64, 64
    pos_emb_interpolation_mode     bilinear
    merge_type      sd2_tpool      merge_kernel_size   [2, 2]
    mm_projector_type patchmergerv2  mm_hidden_size    1024
    projector_hidden_act gelu       projector_ln_eps   1e-05
    text_hidden_size       7168

Two things are inferences rather than transcriptions, flagged because they are
the places this could be wrong:

* ``qkv_hidden_size`` (1536) is wider than ``vt_hidden_size`` (1024), giving
  head_dim = 1536 / 12 = 128 and an output projection that CONTRACTS 1536 ->
  1024. That is unusual for a ViT but is the only reading consistent with both
  fields.
* The report says attention "is factorized into intra-frame spatial and
  inter-frame temporal passes" and that the tower is "roughly 0.4B parameters".
  Those two only reconcile if the spatial and temporal passes SHARE one set of
  qkv/out projections: 27 x (3*1024*1536 + 1536*1024 + 2*1024*4096) = 396M,
  whereas separate per-pass projections would give 566M. So the passes here
  share parameters and differ only in which axis they attend over. See
  ``test_moonvit.py::test_parameter_count_matches_the_reported_0p4b``.

Shape suffixes: B batch, F frames, H patch rows, W patch cols, N tokens per
frame (H*W), D model dim (vt_hidden_size), Q qkv_hidden_size, A heads,
K head_dim, C text_hidden_size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(kw_only=True)
class MoonViTConfig:
    """MoonViT-V2 config. Defaults are K3's released values."""

    num_hidden_layers: int = 27
    hidden_size: int = 1024
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    intermediate_size: int = 4096
    patch_size: int = 14
    in_channels: int = 3
    rms_norm_eps: float = 1e-5
    # Learned factorized position tables, interpolated to the real grid.
    init_pos_emb_time: int = 4
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    pos_emb_interpolation_mode: str = "bilinear"
    # Token merging before projection: 2x2 pixel shuffle + temporal pooling.
    merge_kernel_size: tuple[int, int] = (2, 2)
    temporal_pool_size: int = 2
    # Projector target width (the LLM's hidden size).
    text_hidden_size: int = 7168
    projector_ln_eps: float = 1e-5
    initializer_range: float = 0.02

    @property
    def head_dim(self) -> int:
        if self.qkv_hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"qkv_hidden_size {self.qkv_hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        return self.qkv_hidden_size // self.num_attention_heads


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """``gelu_pytorch_tanh``: the tanh approximation, not the erf form."""
    return F.gelu(x, approximate="tanh")


class MoonViTPatchEmbed(nn.Module):
    """Non-overlapping patch projection, no bias.

    A Conv2d with stride == kernel is the same linear map as flattening each
    patch, but keeps the spatial layout explicit for the position tables.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, pixels_BFCHW: torch.Tensor) -> torch.Tensor:
        """``[B, F, C, H, W]`` pixels -> ``[B, F, H/p, W/p, D]`` patches."""
        B, Fr = pixels_BFCHW.shape[:2]
        x = self.proj(pixels_BFCHW.flatten(0, 1))  # [B*F, D, H/p, W/p]
        x = x.permute(0, 2, 3, 1)  # [B*F, H/p, W/p, D]
        return x.unflatten(0, (B, Fr))


class MoonViTPositionEmbedding(nn.Module):
    """``divided_fixed``: separate time / height / width tables, summed.

    "Divided" because the position signal factorizes over the three axes rather
    than being a single joint table; "fixed" because the tables are learned at a
    FIXED grid (4 x 64 x 64) and bilinearly interpolated to whatever resolution
    an input actually has. That is what lets one set of weights serve inputs
    from a single small image up to 3584 x 3584 (256 x 256 patches at
    patch_size 14) without retraining or padding to a canonical size.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.mode = config.pos_emb_interpolation_mode
        D = config.hidden_size
        self.time = nn.Parameter(torch.empty(config.init_pos_emb_time, D))
        # Height and width are interpolated together as one 2-D grid, so a
        # non-square target keeps the aspect handling of a true 2-D resize
        # rather than two independent 1-D stretches.
        self.spatial = nn.Parameter(
            torch.empty(D, config.init_pos_emb_height, config.init_pos_emb_width)
        )

    def forward(self, patches_BFHWD: torch.Tensor) -> torch.Tensor:
        B, Fr, H, W, D = patches_BFHWD.shape
        spatial = self.spatial
        if (H, W) != spatial.shape[1:]:
            spatial = F.interpolate(
                spatial.unsqueeze(0).float(),
                size=(H, W),
                mode=self.mode,
                align_corners=False,
            ).squeeze(0).to(spatial.dtype)
        pos = spatial.permute(1, 2, 0)  # [H, W, D]

        time = self.time
        if Fr != time.shape[0]:
            # 1-D resize over the time axis; a single frame takes table entry 0
            # so an image is not an interpolation of the video tables.
            if Fr == 1:
                time = time[:1]
            else:
                time = F.interpolate(
                    time.t().unsqueeze(0).float(),
                    size=Fr,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).t().to(time.dtype)
        return patches_BFHWD + pos.view(1, 1, H, W, D) + time.view(1, Fr, 1, 1, D)


class MoonViTAttention(nn.Module):
    """Factorized space-time attention with SHARED projections.

    One qkv/out projection set is applied twice per block: once over the tokens
    within each frame (spatial), once over frames at each spatial position
    (temporal). Sharing is what makes the reported 0.4B add up -- see the module
    docstring. For a single frame the temporal pass is over length 1, which is
    the identity up to the projections, so it is skipped.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.qkv_hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            config.qkv_hidden_size, config.hidden_size, bias=False
        )

    def _attend(self, x_BTD: torch.Tensor) -> torch.Tensor:
        """Bidirectional attention over dim 1 of a ``[B', T, D]`` tensor."""
        B, T, _ = x_BTD.shape
        qkv = self.qkv_proj(x_BTD)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (B, T, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        # No causal mask: a vision encoder sees the whole image.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out)

    def forward(self, x_BFHWD: torch.Tensor) -> torch.Tensor:
        B, Fr, H, W, D = x_BFHWD.shape
        # Spatial: each frame independently, over its H*W tokens.
        spatial = self._attend(x_BFHWD.reshape(B * Fr, H * W, D))
        x = spatial.view(B, Fr, H, W, D)
        if Fr == 1:
            return x
        # Temporal: each spatial position independently, over its F frames.
        temporal_in = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, Fr, D)
        temporal = self._attend(temporal_in)
        return temporal.view(B, H, W, Fr, D).permute(0, 3, 1, 2, 4)


class MoonViTMLP(nn.Module):
    """``mlp2``: two layers, gelu_pytorch_tanh, no bias."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_gelu_tanh(self.fc1(x)))


class MoonViTBlock(nn.Module):
    """Pre-norm block with RMSNorm and no biases (report sec 2.4)."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MoonViTAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MoonViTMLP(config)

    def forward(self, x_BFHWD: torch.Tensor) -> torch.Tensor:
        x_BFHWD = x_BFHWD + self.attn(self.attn_norm(x_BFHWD))
        return x_BFHWD + self.mlp(self.mlp_norm(x_BFHWD))


class PatchMergerV2(nn.Module):
    """``patchmergerv2`` + ``sd2_tpool``: 2x2 pixel shuffle, temporal pool, MLP.

    The pixel shuffle is space-to-depth, not pooling: a 2x2 neighbourhood
    becomes 4D channels, so nothing is averaged away before the projector sees
    it. That is the 4x token reduction the report cites for keeping 3584x3584
    inputs affordable in a 1M context. Temporal pooling then averages adjacent
    frames, which does discard information -- it is the cheaper axis to
    compress because adjacent video frames are highly redundant.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        kh, kw = config.merge_kernel_size
        self.kh, self.kw = kh, kw
        self.temporal_pool_size = config.temporal_pool_size
        merged = config.hidden_size * kh * kw
        self.norm = nn.LayerNorm(merged, eps=config.projector_ln_eps)
        self.fc1 = nn.Linear(merged, merged, bias=False)
        self.fc2 = nn.Linear(merged, config.text_hidden_size, bias=False)

    def forward(self, x_BFHWD: torch.Tensor) -> torch.Tensor:
        """``[B, F, H, W, D]`` -> ``[B, F', H/kh * W/kw, text_hidden]``."""
        B, Fr, H, W, D = x_BFHWD.shape
        if H % self.kh or W % self.kw:
            raise ValueError(
                f"patch grid {H}x{W} must divide the merge kernel "
                f"{self.kh}x{self.kw}; pad or resize the input"
            )
        # Space-to-depth: [B, F, H/kh, kh, W/kw, kw, D] -> channels last.
        x = x_BFHWD.view(B, Fr, H // self.kh, self.kh, W // self.kw, self.kw, D)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            B, Fr, (H // self.kh) * (W // self.kw), self.kh * self.kw * D
        )
        if Fr > 1 and self.temporal_pool_size > 1:
            pool = min(self.temporal_pool_size, Fr)
            usable = (Fr // pool) * pool
            if usable:
                pooled = x[:, :usable].view(B, usable // pool, pool, *x.shape[2:])
                pooled = pooled.mean(dim=2)
                # A trailing partial group is kept unpooled rather than dropped:
                # silently discarding frames would change the token count in a
                # way the caller cannot see.
                x = (
                    pooled
                    if usable == Fr
                    else torch.cat([pooled, x[:, usable:]], dim=1)
                )
        return self.fc2(_gelu_tanh(self.fc1(self.norm(x))))


class MoonViT(nn.Module):
    """MoonViT-V2 encoder + PatchMergerV2 projector.

    Images and videos share every parameter, as the report specifies: an image
    is a 1-frame video, which takes the time table's first entry and skips the
    temporal attention pass and the temporal pool.
    """

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = MoonViTPatchEmbed(config)
        self.pos_emb = MoonViTPositionEmbedding(config)
        self.layers = nn.ModuleList(
            MoonViTBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.projector = PatchMergerV2(config)

    def forward(self, pixels_BFCHW: torch.Tensor) -> torch.Tensor:
        """``[B, F, C, H, W]`` pixels -> ``[B, F', N', text_hidden_size]``.

        Accepts ``[B, C, H, W]`` too, treating it as a single frame.
        """
        if pixels_BFCHW.dim() == 4:
            pixels_BFCHW = pixels_BFCHW.unsqueeze(1)
        x = self.patch_embed(pixels_BFCHW)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x)
        return self.projector(self.norm(x))

    def encoder_num_parameters(self) -> int:
        """Parameters in the tower proper, excluding the projector.

        The report's "roughly 0.4B" is the encoder; the projector is called out
        separately as "a lightweight MLP projector", and at text_hidden_size
        7168 it is not lightweight relative to a 0.4B tower, so counting it in
        would make the figure unrecognizable.
        """
        proj = set(id(p) for p in self.projector.parameters())
        return sum(p.numel() for p in self.parameters() if id(p) not in proj)

    def init_weights(self, init_range: float | None = None) -> None:
        """Exhaustive init, for the same meta-device reason as the LLM's."""
        std = init_range if init_range is not None else self.config.initializer_range
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.RMSNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_emb.time, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb.spatial, mean=0.0, std=std)
