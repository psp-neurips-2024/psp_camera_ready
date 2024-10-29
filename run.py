import argparse
import collections
import functools
import yaml
import os
import pathlib
import sys

import gym
from gym.envs import registration
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

from tia.Dreamer.dreamers import Dreamer, SeparationDreamer, InverseDreamer
from tia.Dreamer import env_tools
from tia.Dreamer import tools
from tia.Dreamer import wrappers as tia_wrappers
from wrappers import color_grid_utils

METHOD2DREAMER = {
    'dreamer': Dreamer,
    'tia': SeparationDreamer,
    'inverse': InverseDreamer
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
tf.get_logger().setLevel('ERROR')
sys.path.append(str(pathlib.Path(__file__).parent))


def make_env(config, writer, prefix, datadir, store):
    suite, domain_task = config.task.split('_', 1)
    domain_name, task_name = domain_task.split('_', 1)

    # Like tia.Dreamer.dmc2gym.make.
    env_id = 'dmc_%s_%s_%s' % (domain_name, task_name, config.seed)
    env_id += f'{"-".join(map(str, config.action_dims_to_split))}_{config.num_cells_per_dim}_{config.num_colors_per_cell}_{config.evil_level}-v1'

    evil_level = color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING[config.evil_level]
    if env_id not in gym.envs.registry:
        registration.register(
            id=env_id,
            entry_point='wrappers.color_grid:DmcColorGridWrapper',
            kwargs={
                'domain_name': domain_name,
                'task_name': task_name,
                'task_kwargs': {
                    'random': config.seed
                },
                'num_cells_per_dim': config.num_cells_per_dim,
                'num_colors_per_cell': config.num_colors_per_cell,
                'evil_level': evil_level,
                'action_dims_to_split': config.action_dims_to_split,
                'action_power': config.action_power,
                'action_splits': (
                    config.action_splits
                    if evil_level != color_grid_utils.EvilEnum.NATURAL
                    else None),
                'natural_video_dir': config.natural_video_dir,
                'total_natural_frames': config.total_natural_frames,
                'environment_kwargs': None,
                'visualize_reward': False,
                'from_pixels': True,
                'height': config.image_size,
                'width': config.image_size,
                'camera_id': 0,
                'frame_skip': config.action_repeat,
                'bg_color': ([int(i) for i in config.bg_color]
                             if config.bg_color else None),
                'no_agent': config.no_agent,
            },
        )
    env = gym.make(env_id)
    env = tia_wrappers.DMC2GYMWrapper(env)
    env = tia_wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
    callbacks = []
    if store:
        callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
    callbacks.append(lambda ep: env_tools.summarize_episode(
        ep, config, datadir, writer, prefix))
    env = tia_wrappers.Collect(env, callbacks, config.precision)
    env = tia_wrappers.RewardObs(env)
    return env



def main(config):
    if config.method == 'separation':
        config.logdir = os.path.join(
            config.logdir, config.task,
            'separation' + '_' + str(config.disen_neg_rew_scale) +
            '_' + str(config.disen_rec_scale),
            str(config.seed))
    else:
        config.logdir = os.path.join(
            config.logdir, config.task,
            args.method,
            str(config.seed))

    logdir = pathlib.Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = os.path.join(config.logdir, 'snapshots')
    snapshot_dir = pathlib.Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.logdir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(config), f, sort_keys=False)

    if config.gpu_growth:
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        prec.set_global_policy(prec.Policy('mixed_float16'))
    config.steps = int(config.steps)
    config.logdir = logdir
    if config.debug:
        tf.config.experimental_run_functions_eagerly(True)
    print('Logdir', config.logdir)

    # Create environments.
    datadir = config.logdir / 'episodes'
    writer = tf.summary.create_file_writer(
        str(config.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    train_envs = [tia_wrappers.Async(lambda: make_env(
        config, writer, 'train', datadir, store=True), config.parallel)
        for _ in range(config.envs)]
    test_envs = [tia_wrappers.Async(lambda: make_env(
        config, writer, 'test', datadir, store=False), config.parallel)
        for _ in range(config.envs)]
    actspace = train_envs[0].action_space

    # Prefill dataset with random episodes.
    step = env_tools.count_steps(datadir, config)
    prefill = max(0, config.prefill - step)
    print(f'Prefill dataset with {prefill} steps.')

    def random_agent(o, d, _):
        return ([actspace.sample() for _ in d], None)

    tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
    writer.flush()

    # Train and regularly evaluate the agent.
    step = env_tools.count_steps(datadir, config)
    print(f'Simulating agent for {config.steps - step} steps.')
    DreamerModel = METHOD2DREAMER[config.method]
    agent = DreamerModel(config, datadir, actspace, writer)
    if (config.logdir / 'variables.pkl').exists():
        print('Load checkpoint.')
        agent.load(config.logdir / 'variables.pkl')
    state = None
    should_snapshot = tools.Every(config.snapshot_every)
    while step < config.steps:
        print('Start evaluation.')
        tools.simulate(
            functools.partial(agent, training=False), test_envs, episodes=1)
        writer.flush()
        print('Start collection.')
        steps = config.eval_every // config.action_repeat
        state = tools.simulate(agent, train_envs, steps, state=state)
        step = env_tools.count_steps(datadir, config)
        agent.save(config.logdir / 'variables.pkl')
        if should_snapshot(step):
            agent.save(snapshot_dir / ('variables_' + str(step) + '.pkl'))
    for env in train_envs + test_envs:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', type=str, choices=['dreamer', 'inverse', 'tia'],
        required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument(
        '--action_dims_to_split', nargs='+', type=int, required=True)
    parser.add_argument('--num_cells_per_dim', type=int, required=True)
    parser.add_argument('--num_colors_per_cell', type=int, required=True)
    parser.add_argument(
        '--evil_level',
        choices=list(color_grid_utils.EVIL_CHOICE_CONVENIENCE_MAPPING.keys()),
        required=True)
    parser.add_argument(
        '--action_power', type=int, required=True
    )
    parser.add_argument('--action_splits', nargs='*', type=int)
    parser.add_argument('--no_agent', action='store_true')
    parser.add_argument('--agent', dest='no_agent', action='store_false')
    parser.add_argument('--natural_video_dir', type=str)
    parser.add_argument('--total_natural_frames', type=int)
    parser.set_defaults(no_agent=False)
    args, remaining = parser.parse_known_args()
    config_path = 'tia/Dreamer/train_configs/' + args.method + '.yaml'
    configs = yaml.safe_load(
        (pathlib.Path(__file__).parent / config_path).read_text())
    config_ = {}
    for name in args.configs:
        config_.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in config_.items():
        parser.add_argument(
            f'--{key}', type=tools.args_type(value), default=value)
    all_args = vars(args)
    if all_args['action_splits']:
        all_args['action_power'] = None
    all_args.update(vars(parser.parse_args(remaining)))
    class Config:
        def __init__(self, all_args):
            for k, v in all_args.items():
                setattr(self, k, v)

    main(Config(all_args))
