import os
import threading

import gym
import numpy as np

from . import natural_imgsource

from wrappers import color_grid


class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    ################################################################################
    # Implementation of Cheetah/Walker Run Sparse follows
    # https://github.com/younggyoseo/RE3
    self._name = name
    if name == 'cheetah_run_sparse':
      name = 'cheetah_run'
    if name == 'walker_run_sparse':
      name = 'walker_run'
    ################################################################################
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      ################################################################################
      step_reward = time_step.reward
      if step_reward is not None:
        if self._name in ('cheetah_run_sparse', 'walker_run_sparse'):
          reward += 0 if step_reward < 0.25 else step_reward
        else:
          reward += step_reward
      ################################################################################
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


################################################################################
# Implementation of natural background DMC follows
# https://github.com/VinAIResearch/TPC-tensorflow
class NAT:

  def __init__(
      self, name, action_repeat=1, size=(64, 64), camera=None,
      bg_path=None, random_bg=False, max_videos=100):
    os.environ['MUJOCO_GL'] = 'egl'
    ################################################################################
    # Implementation of Cheetah/Walker Run Sparse follows
    # https://github.com/younggyoseo/RE3
    self._name = name
    if name == 'cheetah_run_sparse':
      name = 'cheetah_run'
    if name == 'walker_run_sparse':
      name = 'walker_run'
    ################################################################################
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera
    ################################################################################
    assert bg_path is not None
    files = [
        os.path.join(bg_path, f) for f in os.listdir(bg_path)
        if os.path.isfile(os.path.join(bg_path, f))]
    self._bg_source = natural_imgsource.RandomVideoSource(
        size, files, random_bg, max_videos, grayscale=False)
    ################################################################################

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      ################################################################################
      step_reward = time_step.reward
      if step_reward is not None:
        if self._name in ('cheetah_run_sparse', 'walker_run_sparse'):
          reward += 0 if step_reward < 0.25 else step_reward
        else:
          reward += step_reward
      ################################################################################
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    ################################################################################
    self._bg_source.reset()
    ################################################################################
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    ################################################################################
    obs = self._env.physics.render(*self._size, camera_id=self._camera)
    mask = np.logical_and((obs[:, :, 2] > obs[:, :, 1]), (obs[:, :, 2] > obs[:, :, 0]))
    bg = self._bg_source.get_image()
    obs[mask] = bg[mask]
    return obs
    ################################################################################
################################################################################


class ColorGridBg(color_grid.DmcColorGridWrapper):
  def __init__(
      self, name, action_repeat=1, size=(64, 64),
      num_cells_per_dim=None, num_colors_per_cell=None, evil_level=None,
      action_dims_to_split=[], action_power=2, action_splits=None,
      no_agent=False, task_kwargs=None, visualize_reward={},
      from_pixels=False, natural_video_dir=None, total_natural_frames=1_000):
    domain, task = name.split('_')
    super().__init__(
      domain, task, num_cells_per_dim, num_colors_per_cell, evil_level,
      action_dims_to_split=action_dims_to_split, action_power=action_power,
      action_splits=action_splits if action_power < 0 else None,
      no_agent=no_agent, task_kwargs=task_kwargs,
      visualize_reward=visualize_reward, from_pixels=from_pixels,
      height=size[0], width=size[1], camera_id=0,
      frame_skip=action_repeat,
      natural_video_dir=natural_video_dir,
      total_natural_frames=total_natural_frames)
    self._size = size
    self.observation_space = self._get_observation_space()
    self.action_space = self._get_action_space()

  def _get_observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
        -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
      0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  def _get_action_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def _convert_action(self, action):
    action = action['action']
    assert np.isfinite(action).all(), action
    return action

  def _wrap_obs(self, time_step, obs):
    new_obs = dict(time_step.observation)
    new_obs['image'] = obs
    return new_obs


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky_actions else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
    return gym.spaces.Dict({
        'image': self._env.observation_space,
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    })

  @property
  def action_space(self):
    return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs

  def step(self, action):
    action = action['action']
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs, reward, done, info

  def render(self, mode):
    return self._env.render(mode)


class Dummy:

  def __init__(self):
    pass

  @property
  def observation_space(self):
    image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
    return gym.spaces.Dict({'image': image})

  @property
  def action_space(self):
    action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
    return gym.spaces.Dict({'action': action})

  def step(self, action):
    obs = {'image': np.zeros((64, 64, 3))}
    reward = 0.0
    done = False
    info = {}
    return obs, reward, done, info

  def reset(self):
    obs = {'image': np.zeros((64, 64, 3))}
    return obs


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.action_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig})


class OneHotAction:

  def __init__(self, env, key='action'):
    assert isinstance(env.action_space[key], gym.spaces.Discrete)
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

  def step(self, action):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index})

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class RewardObs:

  def __init__(self, env, key='reward'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class ResetObs:

  def __init__(self, env, key='reset'):
    assert key not in env.observation_space.spaces
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    space = gym.spaces.Box(0, 1, (), dtype=np.bool)
    return gym.spaces.Dict({
        **self._env.observation_space.spaces, self._key: space})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reset'] = np.array(False, np.bool_)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reset'] = np.array(True, np.bool_)
    return obs
