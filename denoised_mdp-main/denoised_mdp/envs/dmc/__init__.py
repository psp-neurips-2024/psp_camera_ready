# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from ..abc import EnvBase

from .dmc2gym.wrappers import DMCWrapper


def make_env(spec: str, observation_output_kind: EnvBase.ObsOutputKind, seed,
             max_episode_length, action_repeat, batch_shape,
             num_cells_per_dim, num_colors_per_cell,
             evil_level, action_dims_to_split,
             action_power,
             action_splits,
             no_agent, natural_video_dir, total_natural_frames):
    # avoid circular imports
    from ..utils import make_batched_auto_reset_env, as_SeedSequence, get_kinetics_dir

    domain_name, task_name = spec.split('_', maxsplit=1)

    def make_kwargs(seed):
        # TODO: Thread color grid kwargs through and plug in here.
        return dict(
            domain_name=domain_name,
            task_name=task_name,
            observation_output_kind=observation_output_kind,
            frame_skip=action_repeat,
            height=64,
            width=64,
            episode_length=max_episode_length,
            # others
            camera_id=0,
            environment_kwargs=None,
            task_kwargs={'random': seed},
            visualize_reward=False,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=evil_level,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            action_splits=action_splits,
            no_agent=no_agent,
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames,
        )

    return make_batched_auto_reset_env(
        lambda seed: DMCWrapper(**make_kwargs(seed)),
        seed, batch_shape)


__all__ = ['make_env']
