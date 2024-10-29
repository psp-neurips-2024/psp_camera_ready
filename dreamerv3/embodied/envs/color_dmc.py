import functools
import os

import numpy as np

from dreamerv3 import embodied
from wrappers import color_grid_utils


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
    locom_rodent=1,
    quadruped=2,
  )

  def __init__(
          self,
          env,
          repeat=1,
          render=True,
          size=(64, 64),
          camera=-1,
          num_cells_per_dim=2,
          num_colors_per_cell=2,
          evil_level=color_grid_utils.EvilEnum.NONE,
          action_dims_to_split=[],
          action_power=None,
          action_splits=None,
          include_foreground_mask=False,
          natural_video_dir=None,
          mask_spaces=None,
          total_natural_frames=1000,
          random_seed=1,
  ):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    domain, task = env.split('_', 1)
    if camera == -1:
      camera = self.DEFAULT_CAMERAS.get(domain, 0)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    else:
      from tia.Dreamer import local_dm_control_suite as suite
      env = suite.load(domain, task)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._render = render
    self._size = size
    self._camera = camera
    self._color_bg = color_grid_utils.ColorGridBackground(
      domain_name=domain,
      task_name=task,
      num_cells_per_dim=num_cells_per_dim,
      num_colors_per_cell=num_colors_per_cell,
      evil_level=evil_level,
      action_dims_to_split=action_dims_to_split,
      action_power=action_power,
      action_splits=action_splits,
      height=size[0],
      width=size[1],
      natural_video_dir=natural_video_dir,
      total_natural_frames=total_natural_frames,
      random_seed=random_seed,
    )
    self._evil_level = evil_level
    self._step = 0
    self._include_foreground_mask = include_foreground_mask
    self._mask_spaces = mask_spaces


  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    if self._mask_spaces:
      spaces.update(self._mask_spaces)
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    if self._render:
      image, mask = self.render(action, obs)
      obs['image'] = image
      if self._include_foreground_mask:
        obs['foreground_mask'] = mask
    return obs

  def render(self, action, obs):
    image = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    if self._evil_level is color_grid_utils.EvilEnum.NONE:
      return image, None

    if action['reset']:
      self._step = 0

    bg_image = self._color_bg.get_background_image(
      self._step, action['action'], obs['reward'])
    mask = np.logical_and(
      (image[:, :, 2] > image[:, :, 1]),
      (image[:, :, 2] > image[:, :, 0])  # hardcoded for dmc
    )
    image[mask] = bg_image[mask]

    if obs['is_last']:
      self._step = 0
    else:
      self._step += 1

    return image, mask
