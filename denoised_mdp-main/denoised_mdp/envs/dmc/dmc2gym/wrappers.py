# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# See also denoised_mdp/envs/dmc/dmc2gym/README.md for license-related
# information about these files adapted from
# https://github.com/facebookresearch/deep_bisim4control/

from typing import *

import glob
import os
import contextlib

import cv2
from gym import core, spaces
from dm_env import specs
import numpy as np
import torch

from . import local_dm_control_suite as suite
from . import natural_imgsource
from ...abc import EnvBase, AutoResetEnvBase, IndexableSized
from ...utils import as_SeedSequence, split_seed
from .... import utils

from wrappers import color_grid


def _spec_to_box(spec, repeat=1):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.tile(np.concatenate(mins, axis=0), repeat).astype(np.float32)
    high = np.tile(np.concatenate(maxs, axis=0), repeat).astype(np.float32)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs, dtype=None):
    obs_pieces = []
    for v in obs.values():
        if np.isscalar(v):
            flat = np.array([v], dtype=dtype)
        else:
            flat = np.asarray(v.ravel(), dtype=dtype)
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env, EnvBase):
    @property
    def max_episode_length(self) -> int:
        return self._episode_length

    @property
    def observation_output_kind(self) -> EnvBase.ObsOutputKind:
        return self._observation_output_kind

    @property
    def action_repeat(self) -> int:
        return self._frame_skip

    def __init__(
        self,
        domain_name,
        task_name,
        num_cells_per_dim,
        num_colors_per_cell,
        evil_level,
        natural_video_dir,
        total_natural_frames,
        action_dims_to_split=[],
        action_power=2,
        action_splits=None,
        no_agent=False,
        task_kwargs={},
        visualize_reward={},
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        observation_output_kind=EnvBase.ObsOutputKind.image_uint8,
        episode_length=1000,
    ):
        # TODO: Stop accessing protected attributes on this.
        self.cg_wrapper = color_grid.DmcColorGridWrapper(
            domain_name,
            task_name,
            num_cells_per_dim,
            num_colors_per_cell,
            evil_level,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            action_splits=action_splits,
            no_agent=no_agent,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            from_pixels=True,
            height=height,
            width=width,
            camera_id=camera_id,
            frame_skip=frame_skip,
            environment_kwargs=environment_kwargs,
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames,
        )

        self._observation_output_kind = observation_output_kind

        self._episode_length = episode_length

        # create observation space
        self._observation_space = EnvBase.ObsOutputKind.get_observation_space(
            observation_output_kind, height, width)

        self._steps_taken = 0

    # def __getattr__(self, name):
    #     return getattr(self.cg_wrapper._env, name)

    def _get_obs(self, timestep, action, reward):
        obs = self.cg_wrapper._get_obs(timestep, action, reward)
        obs = self.ndarray_uint8_image_to_observation(obs, target_shape=None)

        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self.cg_wrapper._true_action_space.high - self.cg_wrapper._true_action_space.low
        norm_delta = self.cg_wrapper._norm_action_space.high - self.cg_wrapper._norm_action_space.low
        action = (action - self.cg_wrapper._norm_action_space.low) / norm_delta
        action = action * true_delta + self.cg_wrapper._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def internal_state_space(self):
        return self.cg_wrapper._internal_state_space

    @property
    def action_space(self):
        return self.cg_wrapper._norm_action_space

    def sample_random_action(self, size=(), np_rng=None):
        if np_rng is None:
            np_rng = np.random
        return torch.as_tensor(np_rng.uniform(-1, 1, size=tuple(size) + tuple(self.action_shape)), dtype=torch.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach()
        action = np.asarray(action, dtype=np.float32)
        obs, reward, done, extra = self.cg_wrapper.step(self._convert_action(action))
        obs = self.ndarray_uint8_image_to_observation(obs, target_shape=None)
        return obs, reward, done, EnvBase.Info(extra['actual_env_steps_taken'])

    def reset(self) -> Tuple[torch.Tensor, EnvBase.Info]:
        return (
            self.ndarray_uint8_image_to_observation(self.cg_wrapper.reset()),
            EnvBase.Info(0))

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        return self.cg_wrapper.render(
            mode=mode, height=height, width=width, camera_id=camera_id)
