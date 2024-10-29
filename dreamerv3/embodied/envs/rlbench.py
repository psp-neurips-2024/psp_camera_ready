import functools

from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from dreamerv3 import embodied

import numpy as np
from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

# https://github.com/younggyoseo/MWM/blob/8394788bf4dcdc94c8b8ad1afb6ee36741716620/mwm/common/envs.py#L269
class RLBench(embodied.Env):
  def __init__(
      self,
      name,
      size=(64, 64,),
      action_repeat=1,
      shadows=True,
      max_length=10_000
  ):
    # we only support reach_target in this codebase
    obs_config = ObservationConfig()
    obs_config.left_shoulder_camera.set_all(False)
    obs_config.right_shoulder_camera.set_all(False)
    obs_config.overhead_camera.set_all(False)
    obs_config.wrist_camera.set_all(False)
    obs_config.front_camera.image_size = size
    obs_config.front_camera.depth = False
    obs_config.front_camera.point_cloud = False
    obs_config.front_camera.mask = False
    obs_config.front_camera.render_mode = (
      RenderMode.OPENGL3 if shadows else RenderMode.OPENGL
    )

    action_mode = functools.partial(JointPosition, absolute_mode=False)

    env = Environment(
      action_mode=MoveArmThenGripper(
        arm_action_mode=action_mode(), gripper_action_mode=Discrete()
      ),
      obs_config=obs_config,
      headless=True,
      shaped_rewards=True,
    )
    env.launch()

    if name == 'reach_target':
      task = ReachTarget
    else:
      raise ValueError(
        f'{name} not supported by RLBench env, only reach_target is '
        f'currently supported.')
    self._env = env
    self._task = env.get_task(task)

    _, obs = self._task.reset()
    self._prev_obs = None

    self._size = size
    self._action_repeat = action_repeat
    self._step = 0
    self._max_length = max_length

  @functools.cached_property
  def obs_space(self):
    spaces = {
      'image': embodied.Space(np.uint8, self._size + (3,)),
      'reward': embodied.Space(np.float32, ()),
      'is_first': embodied.Space(np.bool),
      'is_last': embodied.Space(np.bool),
      'is_terminal': embodied.Space(np.bool),
      'success': embodied.Space(np.bool),
    }
    return spaces

  @functools.cached_property
  def act_space(self):
    action = embodied.Space(
      np.float32,
      tuple(int(i) for i in self._env.action_shape),
      -1., 1.)
    return {'action': action}

  def step(self, action):
    assert np.isfinite(action['action']).all(), action['action']
    if action['reset']:
      return self._reset()
    try:
      terminal = True
      success = False
      reward = 0.0
      obs = self._prev_obs
      for i in range(self._action_repeat):
        obs, reward_, terminal = self._task.step(action['action'])
        self._step += 1
        success, _ = self._task._task.success()
        reward += reward_
        if terminal:
          break
      self._prev_obs = obs
    except (IKError, ConfigurationPathError, InvalidActionError) as e:
      terminal = True
      success = False
      reward = 0.0
      obs = self._prev_obs

    obs = {
      'reward': reward,
      'is_first': False,
      'is_last': terminal or self._step >= self._max_length,
      'is_terminal': terminal,
      'image': obs.front_rgb,
      'success': success,
    }
    return obs

  def _reset(self):
    _, obs = self._task.reset()
    self._prev_obs = obs
    self._step = 0
    obs = {
      'reward': 0.0,
      'is_first': True,
      'is_last': False,
      'is_terminal': False,
      'image': obs.front_rgb,
      'success': False
    }
    return obs


