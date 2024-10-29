# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import *

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Protocol

import attrs
import enum
from omegaconf import MISSING

from .abc import AutoResetEnvBase
from .utils import split_seed
from .interaction import env_interact_random_actor, env_interact_with_model, EnvInteractData

from wrappers import color_grid_utils


class EnvKind(enum.Enum):
    dmc = enum.auto()
    robodesk = enum.auto()
    old_robodesk = enum.auto()

    @staticmethod
    def create(kind,
               spec: str,
               action_repeat: int,
               max_episode_length: int,
               num_cells_per_dim: int,
               num_colors_per_cell: int,
               evil_level: str,
               action_dims_to_split: List[int],
               action_power: int,
               action_splits: List[int],
               no_agent: bool,
               natural_video_dir: str,
               total_natural_frames: int,
               *, for_storage: bool,
               seed: int,
               batch_shape=()) -> AutoResetEnvBase:
        if kind is EnvKind.dmc:
            from .dmc import make_env
        else:
            raise ValueError('oopsie whoopsie')

        observation_output_kind: AutoResetEnvBase.ObsOutputKind
        if for_storage:
            observation_output_kind = AutoResetEnvBase.ObsOutputKind.image_uint8
        else:
            observation_output_kind = AutoResetEnvBase.ObsOutputKind.image_float32
        return make_env(
            spec,
            observation_output_kind=observation_output_kind,
            action_repeat=action_repeat,
            max_episode_length=max_episode_length,
            # TODO: Populate seed properly.
            seed=42,
            batch_shape=batch_shape,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING[evil_level],
            action_dims_to_split=action_dims_to_split,
            action_power=action_power if not action_splits else None,
            action_splits=action_splits if action_splits else None,
            no_agent=no_agent,
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames,
        )



@attrs.define(kw_only=True, auto_attribs=True)
class EnvConfig:
    _target_: str = attrs.Factory(lambda: f"{EnvKind.create.__module__}.{EnvKind.create.__qualname__}")
    _partial_: bool = True

    class InstantiatedT(Protocol):
        def __call__(self, *, for_storage: bool, seed: int, batch_shape=()) -> AutoResetEnvBase: ...

    kind: EnvKind = MISSING
    spec: str = MISSING
    action_repeat: int = attrs.field(default=2, validator=attrs.validators.gt(0))
    max_episode_length: int = attrs.field(default=1000, validator=attrs.validators.gt(0))
    num_cells_per_dim: int = attrs.field(default=16, validator=attrs.validators.gt(0))
    num_colors_per_cell: int = attrs.field(default=11664, validator=attrs.validators.gt(0))
    evil_level: str = attrs.field(default='max')
    action_dims_to_split: List[int] = attrs.field(default=[0, 1, 2, 3, 4, 5])
    action_power: int = attrs.field(default=3, validator=attrs.validators.gt(0))
    action_splits: List[int] = attrs.field(default=[])
    no_agent: bool = attrs.field(default=False)
    natural_video_dir: str = MISSING
    total_natural_frames: int = MISSING



__all__ = ['AutoResetEnvBase', 'split_seed', 'env_interact_random_actor', 'env_interact_with_model',
           'EnvInteractData']
