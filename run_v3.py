import warnings

import numpy as np

import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import color_dmc
from dreamerv3.embodied.envs import from_gym
from wrappers import color_grid_utils


def main(argv=None):
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['dmc_vision'])
    for name in parsed.configs:
        if name == 'defaults':
            continue
        config = config.update(dreamerv3.configs[name])
    config = embodied.Flags(config).parse(other)
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    print(config)

    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])

    if config.environment == 'dmc':
        if config.image_v_grad_mask_level:
            mask_spaces = {
                'masks': embodied.Space(np.bool_, (500,) + config.env.dmc.size),
                'masks_count': embodied.Space(np.int32, ()),
            }
        else:
            mask_spaces = None
        env = color_dmc.DMC(
            config.task,
            repeat=config.env.dmc.repeat,
            size=config.env.dmc.size,
            camera=config.env.dmc.camera,
            num_cells_per_dim=config.evil.num_cells_per_dim,
            num_colors_per_cell=config.evil.num_colors_per_cell,
            evil_level=color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING[
                config.evil.evil_level
            ],
            action_dims_to_split=config.evil.action_dims_to_split,
            action_power=(
                config.evil.action_power if config.evil.action_power >= 0
                else None),
            action_splits=(
                config.evil.action_splits if config.evil.action_power < 0
                else None),
            natural_video_dir=config.evil.natural_video_dir,
            mask_spaces=mask_spaces,
            total_natural_frames=config.evil.total_natural_frames,
            random_seed=config.evil.random_seed
        )
    elif config.environment == 'rlbench':
        # TODO: Add support for mask spaces to RLBench as well.
        from dreamerv3.embodied.envs import rlbench
        env = rlbench.RLBench(
            config.task,
            size=config.env.rlbench.size,
            action_repeat=config.env.rlbench.repeat,
            shadows=config.env.rlbench.shadows,
            max_length=config.env.rlbench.max_length
        )
    else:
        raise ValueError(f'{config.environment} not supported.')
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

    if config.seg_with_sam:
        replay = embodied.replay.UniformProcessed(
            config.batch_length, config.replay_size,
            logdir / 'preprocessed_replay',
            logdir / 'postprocessed_replay',
            config.max_chunks_behind)
    else:
        replay = embodied.replay.Uniform(
            config.batch_length, config.replay_size, logdir / 'replay')

    embodied.run.train(agent, env, replay, logger, args, config)


if __name__ == '__main__':
    main()