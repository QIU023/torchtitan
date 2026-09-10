# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shared NaViT resize against the released Kimi K3 media processor.

The released processor (``media_utils.navit_resize_image`` in the checkpoint
directory named by ``KIMI_K3_RELEASED_DEBUG_DIR``) decides the resized size,
the padding and the token count with integer arithmetic, so the comparison is
exact: a disagreement changes the patch grid while every downstream shape check
still passes. Skipped when the released processor is not on the machine.
"""

import importlib.util
import json
import os
import pathlib

import pytest

from torchtitan.hf_datasets.multimodal.utils.image import resize_to_navit_patch_grid

_RELEASED_DIR = pathlib.Path(
    os.environ.get("KIMI_K3_RELEASED_DEBUG_DIR", "/workspace/k3qat_mm_hf")
)


def _released_media_utils():
    path = _RELEASED_DIR / "media_utils.py"
    if not path.exists():
        pytest.skip(f"no released media processor at {path}")
    spec = importlib.util.spec_from_file_location("kimi_k3_media_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _released_budgets() -> dict:
    path = _RELEASED_DIR / "preprocessor_config.json"
    if not path.exists():
        pytest.skip(f"no released preprocessor config at {path}")
    return json.load(open(path))["media_proc_cfg"]


_SIZES = [
    (28, 28),
    (224, 224),
    (640, 480),
    (1023, 767),
    (1920, 1080),
    (3000, 2000),
    (7300, 100),
    (100, 7300),
    (8192, 8192),
    (15, 4000),
]


@pytest.mark.parametrize("width,height", _SIZES)
def test_resize_matches_the_released_processor(width, height):
    media = _released_media_utils()
    budgets = _released_budgets()
    patch = budgets["patch_size"]
    merge = budgets["merge_kernel_size"]
    try:
        theirs = media.navit_resize_image(
            width,
            height,
            patch_size=patch,
            merge_kernel_size=merge,
            in_patch_limit=budgets["in_patch_limit"],
            patch_limit_on_one_side=budgets["patch_limit_on_one_side"],
            fixed_output_tokens=None,
        )
    except AssertionError:
        pytest.skip("the released processor rejects this size")
    resize_h, resize_w, pad_h, pad_w = resize_to_navit_patch_grid(
        height,
        width,
        patch_size=patch,
        merge_size=merge,
        max_patches=budgets["in_patch_limit"],
        max_patches_per_side=budgets["patch_limit_on_one_side"],
    )
    assert (resize_w, resize_h, pad_w, pad_h) == (
        theirs["new_width"],
        theirs["new_height"],
        theirs["pad_width"],
        theirs["pad_height"],
    )
    factor = patch * merge
    tokens = ((resize_h + pad_h) // factor) * ((resize_w + pad_w) // factor)
    assert tokens == theirs["num_tokens"]
