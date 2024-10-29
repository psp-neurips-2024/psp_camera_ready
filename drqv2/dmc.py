# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation
from tia.Dreamer import local_dm_control_suite as suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

from wrappers import color_grid_utils


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class BackgroundWrapper(dm_env.Environment):
    def __init__(
            self,
            env,
            domain_name,
            task_name,
            num_cells_per_dim,
            num_colors_per_cell,
            evil_level,
            height,
            width,
            natural_video_dir,
            total_natural_frames,
            pixels_key='pixels',
            action_dims_to_split=[],
            action_power=2,
            action_splits=None,
            random_seed=1,
            no_agent=False,
    ):
        """
        Replaces the background of the wrapped DMC environment's observations
        with a dynamic grid of colors determined from other variables of the
        MDP. The specifics are controlled by the arguments to the constructor
        as detailed below.


        :param env: DMC environment being wrapped.
        :param domain_name: DMC domain name.
        :param task_name: DMC task name.
        :param num_cells_per_dim: Number of cells on the horizontal and
            vertical dimensions of the environment. So e.g. 16 begets a 16x16
            grid of colors.
        :param num_colors_per_cell: The number of total colors per cell. A
            mapping of index to color is stored for each cell, which can be
            used by the specified mapping of MDP variables to color grids.
            In most instantiations `evil_level`, one set of MDP variables
            maps to the same index for every cell. In that case, this
            parameter can be more simply considered as the total number of
            backgrounds.
        :param evil_level: The type of mapping that will be generated between
            MDP variables and backgrounds for the DMC environment.

            `MAXIMUM_EVIL`: The cartesian product of discretized action and
                reward spaces will be mapped to different backgrounds
                (cell indices). The following additional arguments must be
                specified:
                    - `action_dims_to_split`
                    - Either `action_power` or `action_splits`, but not both.
                The exact algorithm for discretizing the action space is
                described under `EVIL_ACTION`.
                The number of colors for discretizing the reward space will be
                determined according to the floor division of
                `num_colors_per_cell` and the action space cardinality. The
                algorithm for discretizing the reward space is described under
                `EVIL_REWARD`.

                We enforce that `num_colors_per_cell` matches the product of
                the number of the number of action spaces and the number of
                reward spaces, otherwise there will be backgrounds that are
                never used.
            `EVIL_REWARD`: The reward space will be discretized to match the
                `num_colors_per_cell`. The reward space is considered as a
                set of intervals. Intervals may only be one number (e.g.
                you get 10. reward for hitting a target) or a range (e.g.
                you get a certain reward proportional to your velocity,
                capped by physics to within a certain upper bound). We
                enforce that `num_colors_per_cell` is greater than or equal
                to 2 times the number of range intervals plus the number of
                single number intervals. Additionally, the environment should
                have at least one range interval or more than one single
                number intervals. Otherwise, this degenerates to a single
                static background, which can be specified via other
                `evil_level`s.

                The number of individual spaces under each reward interval is
                determined via assigning each single number interval a single
                space, then dividing the remaining spaces evenly between the
                range intervals. Any leftover spaces are divided round-robin
                between the range intervals.
            `EVIL_ACTION`: The following additional arguments must be
                specified:
                    - `action_dims_to_split`
                    - Either `action_power` or `action_splits`
                `action_dims_to_split` specifies the dimensions of the action
                space to consider for mapping to backgrounds.

                `action_power`, if set, specifies how many discrete spaces
                each action dimension will be split into. The total number
                of backgrounds mapped to will be
                `action_power ** len(action_dims_to_split)`.

                `action_splits`, if set, specifies how many spaces each
                individual action dimension will be split into. The total
                number of backgrounds mapped to will be
                `product(action_splits)`, where
                `product = partial(reduce, lambda x, y: x * y)`.

                We enforce that `num_colors_per_cell` is equal to the number
                of backgrounds specified by `action_dims_to_split` and
                `action_power` or `action_splits`, as less backgrounds would
                result in an out-of-bounds index for some actions, and more
                backgrounds would simply result in unused backgrounds.
            `EVIL_ACTION_CROSS_SEQUENCE`:  If set, crosses action spaces with
                sequence positions to generate mapping to backgrounds.
                Currently only supports `action_power`, and not
                `action_splits`. The number of colors for discretizing the
                sequence space will be determined according to the floor
                division of `num_colors_per_cell` and the action space
                cardinality, such that our action space assignments are
                unique for a given sequence position according to
                actionSpaceAssignmentsForStep =
                globalSpaceActionAssignments[stepPos % numColorsForSequence].

                We enforce that `num_colors_per_cell` is equal to the product
                of the number of action spaces and number of sequence spaces,
                otherwise there will be backgrounds that are never used.
            `EVIL_SEQUENCE`: If set, assigns sequence positions to
                backgrounds. If the number of steps goes beyond the number of
                assigned backgrounds, we simply loop to the first background.
            `MINIMUM_EVIL`: A random background is chosen on every step.
            `RANDOM`: Each cell's color index is chosen randomly on every step.
                Since cells are not chosen jointly, unlike other modes, this
                is myuch like static, but the colors for each cell are chosen
                in advance.
            `NONE`: The background is not replaced.
        :param height: Image height.
        :param width: Image width.
        :param pixels_key: Key for pixels in `time_step.observation`.
        :param action_dims_to_split: Described under `evil_level`
            `EVIL_ACTION`. Action dimensions to be considered for action to
            background mapping.
        :param action_power: Described under `evil_level`
            `EVIL_ACTION`. Number of spaces to divide each selected action
            dimension into.
        :param action_splits: Described under `evil_level`
            `EVIL_ACTION`. Specifies how many spaces each individual action
            dimension will be split into.
        :param random_seed: Random seed to use for choosing background.
        :param no_agent: The cell colors replace the entire observation image,
            instead of only replacing the background.
        """
        self._env = env
        self._pixels_key = pixels_key
        self._no_agent = no_agent
        self._evil_level = evil_level
        self._num_steps_taken = 0
        self.color_bg = color_grid_utils.ColorGridBackground(
            domain_name=domain_name,
            task_name=task_name,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=evil_level,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            action_splits=action_splits,
            height=height,
            width=width,
            random_seed=random_seed,
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames,
        )

    def reset(self):
        self._num_steps_taken = 0
        time_step = self._env.reset()
        return self._add_background_image(time_step, None)

    def step(self, action):
        time_step = self._env.step(action)
        return self._add_background_image(time_step, action)

    def _add_background_image(self, time_step, action):
        if self._evil_level is color_grid_utils.EvilEnum.NONE:
            return time_step

        bg_image = self.color_bg.get_background_image(
            self._num_steps_taken, action, time_step.reward)
        self._num_steps_taken += 1

        if self._no_agent:
            image = bg_image
        else:
            image = time_step.observation[self._pixels_key].copy()
            # remove batch dim
            if len(image.shape) == 4:
                image = image[0]
            mask = np.logical_and(
                (image[:, :, 2] > image[:, :, 1]),
                (image[:, :, 2] > image[:, :, 0])  # hardcoded for dmc
            )
            image[mask] = bg_image[mask]

        obs = time_step.observation.copy()
        obs[self._pixels_key] = image
        return time_step._replace(observation=obs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def make(
        name,
        frame_stack,
        action_repeat,
        seed,
        num_cells_per_dim,
        num_colors_per_cell,
        evil_level,
        action_dims_to_split,
        action_power,
        action_splits,
        natural_video_dir,
        total_natural_frames,
):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)



        env = BackgroundWrapper(
            env, domain, task,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING[evil_level],
            height=render_kwargs['height'],
            width=render_kwargs['width'],
            pixels_key=pixels_key,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            action_splits=action_splits,
            random_seed=seed,
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames
        )
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
