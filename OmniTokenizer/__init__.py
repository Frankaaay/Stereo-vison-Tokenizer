# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from .download import load_transformer, load_vqgan, download
from .lm_transformer import Net2NetTransformer
from .base import VQGAN as VQGAN
from .omnitokenizer import VQGAN as OmniTokenizer_VQGAN
from .omnitokenizer import (
    MonocularOmniTokenizer, OmniTokenizerOutput, PostFusionOmniTokenizer,
    StereoOmniTokenizer, disparity_to_depth,
)


def __getattr__(name):
    if name in {"VideoData", "ImageDataset", "DecordVideoDataset"}:
        from . import data
        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
