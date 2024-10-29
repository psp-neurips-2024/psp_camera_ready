import enum
import glob
import os

import numpy as np



class EvilEnum(enum.Enum):
    MAXIMUM_EVIL = enum.auto()
    EVIL_REWARD = enum.auto()
    EVIL_ACTION = enum.auto()
    EVIL_ACTION_CROSS_SEQUENCE = enum.auto()
    EVIL_SEQUENCE = enum.auto()
    MINIMUM_EVIL = enum.auto()
    NATURAL = enum.auto()
    RANDOM = enum.auto()
    COLOR = enum.auto()
    NONE = enum.auto()


EVIL_CHOICE_CONVENIENCE_MAPPING = {
    'max': EvilEnum.MAXIMUM_EVIL,
    'reward': EvilEnum.EVIL_REWARD,
    'action': EvilEnum.EVIL_ACTION,
    'sequence': EvilEnum.EVIL_SEQUENCE,
    'action_cross_sequence': EvilEnum.EVIL_ACTION_CROSS_SEQUENCE,
    'minimum': EvilEnum.MINIMUM_EVIL,
    'random': EvilEnum.RANDOM,
    'natural': EvilEnum.NATURAL,
    'none': EvilEnum.NONE,
}


def split_action_space(action_dims, num_colors, power):
    # TODO: Add support for more variation in the future.
    assert num_colors == power ** len(action_dims)
    return [power] * len(action_dims)


def split_reward_space(reward_range, num_colors, max_evil):
    num_non_single = sum(isinstance(r, tuple) for r in reward_range)
    num_single = len(reward_range) - num_non_single
    if max_evil:
        assert num_colors >= len(reward_range)
    else:
        assert num_colors >= 2 * num_non_single + num_single
        assert num_non_single > 0 or len(reward_range) > 1

    if num_non_single == 0:
        assert num_colors == len(reward_range)
    colors_per_range = [
        num_colors // num_non_single - num_single
        if isinstance(r, tuple) else 1
        for r in reward_range
    ]
    num_left = num_colors - sum(colors_per_range)
    while num_left != 0:
        for i in range(len(reward_range)):
            if not isinstance(reward_range[i], tuple):
                continue
            colors_per_range[i] += 1
            num_left -= 1
            if num_left == 0:
                break

    return colors_per_range


def get_min_colors_needed(
        evil_level, reward_range, action_dims_to_split, action_power):
    if evil_level is EvilEnum.MAXIMUM_EVIL:
        return len(reward_range) * action_power ** len(action_dims_to_split)
    elif evil_level is EvilEnum.EVIL_REWARD:
        num_non_single = sum(isinstance(r, tuple) for r in reward_range)
        num_single = len(reward_range) - num_non_single
        return 2 * num_non_single + num_single
    elif evil_level is EvilEnum.EVIL_ACTION:
        return action_power ** len(action_dims_to_split)
    elif evil_level is EvilEnum.EVIL_SEQUENCE:
        return 2
    elif evil_level is EvilEnum.EVIL_ACTION_CROSS_SEQUENCE:
        return 2 * action_power ** len(action_dims_to_split)
    elif evil_level is EvilEnum.MINIMUM_EVIL:
        return 2
    elif evil_level is EvilEnum.RANDOM:
        return 2
    elif evil_level is EvilEnum.NONE:
        return 0
    elif evil_level is EvilEnum.COLOR:
        return 0
    else:
        raise ValueError(f'{evil_level} not implemented.')


def get_num_colors_log_message(num_colors_per_cell, extra):
    return (
        f'num_colors_per_cell {num_colors_per_cell} '
        f'cannot be satisfied. {extra} colors remain for each '
        f'cell.')


def get_colors_for_action_and_reward_max_evil(
        num_colors_per_cell, action_dims_to_split, reward_range, action_power):
    colors_for_action = action_power ** len(action_dims_to_split)
    colors_for_reward = num_colors_per_cell // colors_for_action
    total = colors_for_action * colors_for_reward
    # TODO: Test this logic.
    assert total == num_colors_per_cell, get_num_colors_log_message(
        num_colors_per_cell,
        num_colors_per_cell - colors_for_action * colors_for_reward)
    return (
        split_action_space(
            action_dims_to_split, colors_for_action, power=action_power),
        split_reward_space(reward_range, colors_for_reward, True))


def get_colors_for_action_and_sequence_evil(
        num_colors_per_cell, action_dims_to_split,
        action_power, action_splits):
    assert action_power is None or action_splits is None
    # TODO: Allow crossing anything, refactor all the logic in this file.
    if action_power:
        colors_for_action = action_power ** len(action_dims_to_split)
    else:
        colors_for_action = 1
        for s in action_splits:
            colors_for_action *= s
    colors_for_sequence = num_colors_per_cell // colors_for_action
    total = colors_for_action * colors_for_sequence
    # TODO: Test this logic
    assert num_colors_per_cell == total, get_num_colors_log_message(
        num_colors_per_cell,
        num_colors_per_cell - colors_for_action * colors_for_sequence)
    if action_splits is None:
        action_splits = split_action_space(
            action_dims_to_split, colors_for_action, power=action_power)
    return (action_splits, colors_for_sequence)


def get_colors_for_evil_action(
        num_colors_per_cell, action_dims_to_split, power):
    colors_for_action = power ** len(action_dims_to_split)
    assert colors_for_action == num_colors_per_cell, (
        get_num_colors_log_message(
            num_colors_per_cell,
            num_colors_per_cell - colors_for_action))
    return split_action_space(
        action_dims_to_split, num_colors_per_cell, power=power)


def get_reward_idx(reward, reward_range, colors_per_reward_range):
    range_found = False
    start = 0
    for i, (r, num_colors_in_range) in enumerate(zip(
            reward_range, colors_per_reward_range)):
        if isinstance(r, tuple):
            range_min, range_max = r
            if range_min <= reward <= range_max:
                range_found = True
                break
        else:
            if r == reward:
                range_found = True
                break
        start += num_colors_in_range

    if not range_found:
        raise ValueError(f'Unexpected reward {reward}')
    if isinstance(r, tuple):
        num_colors_in_range = colors_per_reward_range[i]
        delta = (range_max - range_min) / num_colors_in_range
        return start + int((reward - range_min) / delta)
    else:
        return start


def get_action_idx(action, action_dims_to_split, colors_per_action_dim):
    action = [
        action_ for i, action_ in enumerate(action)
        if i in action_dims_to_split
    ]

    for action_ in action:
        assert -1 <= action_ <= 1

    pow = 1
    for num_colors in colors_per_action_dim:
        pow *= num_colors
    action_idx = 0

    for num_colors, action_ in zip(colors_per_action_dim, action):
        delta = 2 / num_colors
        action_idx += (
                int((action_ + (1 - 1e-6)) / delta) * (pow // num_colors))
        pow //= num_colors

    return action_idx


def get_background_image_from_color_grid(
        color_grid,
        height,
        width,
        num_cells_per_dim
):
    bg_image = color_grid
    bg_image = np.repeat(
        bg_image, height // num_cells_per_dim, axis=0)
    bg_image = np.concatenate([
        bg_image,
        np.repeat(
            bg_image[-1:, ...],
            height % num_cells_per_dim,
            axis=0)
    ], axis=0)
    bg_image = np.repeat(
        bg_image, width // num_cells_per_dim, axis=1)
    bg_image = np.concatenate([
        bg_image,
        np.repeat(
            bg_image[:, -1:, :],
            width % num_cells_per_dim,
            axis=1)
    ], axis=1)
    return bg_image


class ColorGridBackground:
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
            natural_video_dir=None,
            height=84,
            width=84,
            random_seed=1,
            total_natural_frames=1_000,
    ):
        """
        Creates a ColorGridBackground class that tracks all random color grid
        backgrounds and returns the appropriate background for each MDP
        transition of an environment. The specifics are controlled by the
        arguments to the constructor as detailed below.

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
                If not using `action_splits`, he number of colors for
                discretizing the sequence space will be determined according
                to the floor division of `num_colors_per_cell` and the action
                space cardinality, such that our action space assignments are
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
            `NATURAL`: Currently hacky, allows re-use of this interface
                for one more distraction task. When this is set to true a
                black-and-white video background replaces the background.
                It is not controlled by the usual params.
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
        :param natural_video_dir: Directory containing natural videos.
            Required if `evil_level` is `NATURAL`. Can be populated for other
            `evil_level`s and will simply serve as a no-op.
        :param height: Image height.
        :param width: Image width.
        :param random_seed: Random seed to use for choosing background.
        """
        assert action_splits is None or action_power is None
        assert (
               action_splits is None
               or evil_level is EvilEnum.EVIL_ACTION
               or evil_level is EvilEnum.EVIL_ACTION_CROSS_SEQUENCE)
        assert (
                action_splits is None
                or len(action_splits) == len(action_dims_to_split))

        self._evil_level = evil_level
        self._num_colors_per_cell = num_colors_per_cell
        self._num_cells_per_dim = num_cells_per_dim
        self._action_dims_to_split = action_dims_to_split
        self._action_power = action_power
        self._natural_video_dir = natural_video_dir
        self._natural_video_source = None
        self._color_grid = None
        self._height = height
        self._width = width

        if domain_name == 'cheetah' and task_name == 'run':
            self.reward_range = [(0, 10,)]
        elif domain_name == 'hopper' and task_name == 'stand':
            self.reward_range = [0, (.8, 1)]
        elif (
            (domain_name == 'walker' and task_name == 'run')
            or (domain_name == 'cartpole' and task_name == 'swingup_sparse')
            or (domain_name == 'cartpole' and task_name == 'swingup')):
            self.reward_range = None
            assert self._evil_level != EvilEnum.EVIL_REWARD and self._evil_level != EvilEnum.MAXIMUM_EVIL
        else:
            # TODO: Also support walker run.
            raise ValueError('Other tasks not supported')

        if action_splits is None and self._evil_level is not EvilEnum.NATURAL:
            min_colors_needed = get_min_colors_needed(
                evil_level, self.reward_range, self._action_dims_to_split,
                self._action_power)
            if self._num_colors_per_cell < min_colors_needed:
                raise ValueError(
                    f'{num_colors_per_cell} insufficient for minimum colors '
                    f'needed {min_colors_needed}')

        if evil_level is EvilEnum.MAXIMUM_EVIL:
            self.colors_per_action_dim, self.colors_per_reward_range = (
                get_colors_for_action_and_reward_max_evil(
                    self._num_colors_per_cell,
                    self._action_dims_to_split,
                    self.reward_range,
                    self._action_power
                )
            )
            self.num_action_indices = 1
            for n in self.colors_per_action_dim:
                self.num_action_indices *= n
        elif evil_level is EvilEnum.EVIL_REWARD:
            self.colors_per_reward_range = split_reward_space(
                self.reward_range, self._num_colors_per_cell, False)
        elif evil_level is EvilEnum.EVIL_ACTION:
            if action_splits is not None:
                self.colors_per_action_dim = action_splits
                total = 1
                for i in self.colors_per_action_dim:
                    total *= i
                assert total == num_colors_per_cell
            else:
                self.colors_per_action_dim = get_colors_for_evil_action(
                    self._num_colors_per_cell, self._action_dims_to_split,
                    self._action_power)
        elif evil_level is EvilEnum.EVIL_ACTION_CROSS_SEQUENCE:
            if action_splits is None:
                self.colors_per_action_dim, self.num_colors_for_sequence = (
                    get_colors_for_action_and_sequence_evil(
                        num_colors_per_cell, action_dims_to_split, action_power,
                        action_splits))
            else:
                self.colors_per_action_dim = action_splits
                total = 1
                for i in self.colors_per_action_dim:
                    total *= i
                self.num_colors_for_sequence = num_colors_per_cell // total
                print(f'num_colors_for_sequence: {self.num_colors_for_sequence}')
                assert total * self.num_colors_for_sequence == num_colors_per_cell

        np.random.seed(random_seed)
        if evil_level is not EvilEnum.NATURAL:
            self._color_grid = np.random.randint(255, size=[
                self._num_cells_per_dim, self._num_cells_per_dim,
                self._num_colors_per_cell, 3])
        else:
            assert natural_video_dir is not None
            files = glob.glob(os.path.expanduser(natural_video_dir))
            assert len(files), (
                f'Pattern {natural_video_dir} does not match any files.')
            from tia.Dreamer.dmc2gym import natural_imgsource
            self._natural_video_source = natural_imgsource.RandomVideoSource(
                (self._height, self._width), files, grayscale=True,
                # Hardcoded to episode length following precedent set by TIA
                # and Denoised MDPs.
                total_frames=total_natural_frames
            )

    def get_background_image(self, t, action, reward) -> np.array:
        """Returns appropriate background image for t, action, reward.

        :param t: Step in sequence. Abbreviated as t to avoid conflating with
            Gym wrapper time_step. Generally each observation image that is
            used to create a downstream representation should have a
            monotonically increasing number t for a given run through an
            environment, and t should start from 0 when the environment
            resets.
        :param action: Action just taken. If action and reward are None this
            signals the environment was reset and a randomly chosen background
            is returned.
        :param reward: Reward just received as a result or R(state, action).
            If action and reward are None this signals the environment was
            reset and a randomly chosen background is returned.
        :return: Appropriate background image for tuple generated from
            (t, action, reward) given the ColorGridBackground initialization.
        """
        if self._evil_level is EvilEnum.NATURAL:
            if t == 0:
                self._natural_video_source.reset()
            return self._natural_video_source.get_image()
        if ((action is None and reward is None)
                or self._evil_level is EvilEnum.RANDOM
                or self._evil_level is EvilEnum.COLOR):
            random_idx = np.random.randint(
                self._num_colors_per_cell,
                size=[self._num_cells_per_dim, self._num_cells_per_dim, 1, 1])
            color_grid = np.take_along_axis(
                self._color_grid, random_idx, 2).squeeze(axis=2)
        elif self._evil_level is EvilEnum.MAXIMUM_EVIL:
            reward_idx = get_reward_idx(
                reward, self.reward_range, self.colors_per_reward_range)
            action_idx = get_action_idx(
                action, self._action_dims_to_split, self.colors_per_action_dim)
            color_grid = self._color_grid[
                         :, :,
                         reward_idx * self.num_action_indices + action_idx,
                         :]
        elif self._evil_level is EvilEnum.EVIL_REWARD:
            reward_idx = get_reward_idx(
                reward, self.reward_range, self.colors_per_reward_range)
            color_grid = self._color_grid[:, :, reward_idx, :]
        elif self._evil_level is EvilEnum.EVIL_ACTION:
            action_idx = get_action_idx(
                action, self._action_dims_to_split, self.colors_per_action_dim)
            color_grid = self._color_grid[:, :, action_idx, :]
        elif self._evil_level is EvilEnum.EVIL_SEQUENCE:
            step_idx = t % self._num_colors_per_cell
            color_grid = self._color_grid[:, :, step_idx, :]
        elif self._evil_level is EvilEnum.EVIL_ACTION_CROSS_SEQUENCE:
            action_idx = get_action_idx(
                action, self._action_dims_to_split, self.colors_per_action_dim)
            step_idx = t % self.num_colors_for_sequence
            idx = self.num_colors_for_sequence * action_idx + step_idx
            color_grid = self._color_grid[:, :, idx, :]
        elif self._evil_level is EvilEnum.MINIMUM_EVIL:
            random_idx = np.random.randint(self._num_colors_per_cell)
            color_grid = self._color_grid[:, :, random_idx, :]
        else:
            raise ValueError(f'{self._evil_level} not supported.')

        return get_background_image_from_color_grid(
            color_grid, self._height, self._width, self._num_cells_per_dim)
