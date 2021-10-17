# Copyright (c) Facebook, Inc. and its affiliates.
"""
Model Zoo API for Detectron2: a collection of functions to create common model architectures
listed in `MODEL_ZOO.md <https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md>`_,
and optionally load their pre-trained weights.
"""

from .model_zoo import  get_config_file, get_checkpoint_url, get_config
from detection_checkpoint import DetectionCheckpointer
from .build import build_model
__all__ = ["get_checkpoint_url", "get_config_file", "get_config", "DetectionCheckpointer", "build_model"]
