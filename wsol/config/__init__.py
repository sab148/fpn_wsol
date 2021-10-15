# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable
from .instantiate import instantiate
from .lazy import LazyCall, LazyConfig

__all__ = [
    "CfgNode",
    "get_cfg",
    "global_cfg",
    "set_global_cfg",
    "downgrade_config",
    "upgrade_config",
    "configurable",
    "instantiate",
    "LazyCall",
    "LazyConfig",
]


from wsol.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata
