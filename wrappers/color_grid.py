from gym import spaces
import numpy as np

from tia.Dreamer.dmc2gym import wrappers
import tia.Dreamer.local_dm_control_suite as suite
from wrappers import color_grid_utils


class DmcColorGridWrapper(wrappers.DMCWrapper):
    def __init__(
        self,
        domain_name,
        task_name,
        num_cells_per_dim,
        num_colors_per_cell,
        evil_level,
        action_dims_to_split=[],
        action_power=2,
        action_splits=None,
        bg_color=None,
        no_agent=False,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        episode_length=None,
        natural_video_dir=None,
        total_natural_frames=1_000,
    ):
        """
        Creates a specialized instance of the TIA wrappers.DMCWrapper.
        This version replaces the background of the wrapped DMC environment's
        observations with a dynamic grid of colors determined from other
        variables of the MDP. The specifics are controlled by the arguments to
        the constructor as detailed below.

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
        :param action_dims_to_split: Described under `evil_level`
            `EVIL_ACTION`. Action dimensions to be considered for action to
            background mapping.
        :param action_power: Described under `evil_level`
            `EVIL_ACTION`. Number of spaces to divide each selected action
            dimension into.
        :param action_splits: Described under `evil_level`
            `EVIL_ACTION`. Specifies how many spaces each individual action
            dimension will be split into.
        :param no_agent: The cell colors replace the entire observation image,
            instead of only replacing the background.
        :param task_kwargs: Dict of keyword arguments for the DMC task.
        :param visualize_reward: Argument for the DMC task; if true
            object colors in rendered frames are set to indicate the reward
            at each step.
        :param from_pixels: Whether to create observation space from pixels or
            from underlying observation space of wrapped environment.
        :param height: Image height.
        :param width: Image width.
        :param camera_id: Environment camera id.
        :param frame_skip: How many times to apply action and accumulate
            reward before returning.
        :param environment_kwargs: Dict of keyword arguments for the DMC
            environment.
        :param episode_length: Maximum episode length.
        """
        assert 'random' in task_kwargs, 'please specify a seed, for deterministic behaviour'
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._episode_length = episode_length
        self._evil_level = evil_level
        self._no_agent = no_agent

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = wrappers._spec_to_box([self._env.action_spec()])
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self.color_bg = color_grid_utils.ColorGridBackground(
           domain_name=domain_name,
            task_name=task_name,
            num_cells_per_dim=num_cells_per_dim,
            num_colors_per_cell=num_colors_per_cell,
            evil_level=evil_level,
            action_dims_to_split=action_dims_to_split,
            action_power=action_power,
            action_splits=action_splits,
            bg_color=bg_color,
            height=height,
            width=width,
            random_seed=task_kwargs.get('random', 1),
            natural_video_dir=natural_video_dir,
            total_natural_frames=total_natural_frames
        )

         # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = wrappers._spec_to_box(
                self._env.observation_spec().values()
            )

        self._internal_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self._env.physics.get_state().shape,
            dtype=np.float32
        )

        # set seed
        self.seed(seed=task_kwargs.get('random', 1))

        self.observation_space = self._observation_space
        self.action_space = self._norm_action_space
        self.num_steps_taken = 0

    def _get_obs(self, time_step, action, reward):
        if self._from_pixels:
            assert not (
                    self._no_agent
                    and self._evil_level is color_grid_utils.EvilEnum.NONE)
            if not self._no_agent:
                obs = self.render(
                    height=self._height,
                    width=self._width,
                    camera_id=self._camera_id
                )
                if self._evil_level is not color_grid_utils.EvilEnum.NONE:
                    bg = self._get_background_image(action, reward)
                    mask = np.logical_and(
                        (obs[:, :, 2] > obs[:, :, 1]),
                        (obs[:, :, 2] > obs[:, :, 0]))  # hardcoded for dmc
                    obs[mask] = bg[mask]
                    obs = obs.copy()
            else:
                obs = self._get_background_image(action, reward)
        else:
            obs = wrappers._flatten_obs(time_step.observation)

        return obs

    def reset(self):
        self.num_steps_taken = 0
        return super().reset()

    def _get_background_image(self, action, reward):
        bg_image = self.color_bg.get_background_image(
            self.num_steps_taken, action, reward)
        self.num_steps_taken += 1
        return bg_image
