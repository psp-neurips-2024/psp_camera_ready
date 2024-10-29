# distracting_benchmarks

This repository contains modified versions of the following:
- Denoised MDPs, the original PyTorch implementation of the Denoised MDP paper.
- Dreamer Pro, the original Tensorflow implementation of the Dreamer Pro paper.
- DreamerV3, the original Jax implementation of the Dreamer V3 paper.
- TIA, the original Tensorflow implementation of the Task Informed Abstractions paper.
- DrQv2, the original PyTorch implementation of the DrQv2 paper.

For all models, the implementation has been modified to introduce the new sensory reafferent environment. The core implementation of this environment can be found under wrappers/color_grid_utils.py. The Distracting Control Suite background is also implemented there, to ensure consistency of the video background between different models, which otherwise would have varying implementations. 

The models are invoked via run.py, in the case of TIA, run_v3.py, in the case of experiments with Dreamer V3, scripts within dreamer_pro_main and denoised_mdp-main.py, in the case of their respective models. More details below. run_sam.py runs the Segment Anything Model and has flags which configure it to watch for output from DreamerV3 in one directory, and save it with masks added in another directory.

Because of the different frameworks and software versions used to implement these models, it is not possible to set up a single environment that can train all models and run all scripts. Our recommendation is to set up a different conda environment for each script. If you wish to reduce the number of required environments, we found that Denoised MDPs, DrQv2, and run_sam.py could reasonably share a single conda environment with PyTorch. 

For Denoised MDPs, Dreamer Pro, Dreamer V3, TIA, and DrQV2 the original authors included instructions for installing dependencies that should continue to work. There are some additional dependencies that should be installed given the modifications. Additionally, changes to NumPy since the version of the Distracting Control Suite shared among these models was implemented have made it incompatible with Scikit-video.

- Video read/write dependencies:
    - Use numpy version 1.20.0
    - pip install imageio-ffmpeg==0.4.4
    - conda install ffmpeg
- Dependency for run_sam.py & run_v3.py to see new files:
    - pip install watchdog

To create a dataset for the Distracting Control Suite benchmark, you can use https://github.com/Showmax/kinetics-downloader. 

Example commands for the V3 experiments in the paper are given with cheetah_run. For ablations simply delete flags as appropriate. The exception is the ablation of the value function instead of policy function. In this case change the value of --gradient_weighting_nets to "value_function". Note when running without "image_value_gradient_smooth_with_sam" that you do not need to create the "pre" and "post" replay directories, nor do you need to invoke "run_sam.py". For different environments e.g. hopper_stand simply substitute the task name. Running environments not included in the paper may require small modifications to the assertions in color_grid_utils.py:386. 

- Deepmind Control Suite Benchmark: 
    mkdir /preferred/location/for/logs
    mkdir /preferred/location/for/logs/preprocessed_replay
    mkdir /preferred/location/for/logs/postprocessed_replay

    /path/to/conda/env/for/dreamerv3/bin/python /path/to/repo/run_v3.py  --configs image_value_gradient image_value_gradient_smooth_with_sam no_evil --gradient_weighting_nets policy_function --embed_only_action_adversarial_head True --image_v_grad_interp_value 0.9 --logdir /preferred/location/for/logs --task cheetah_run --jax.policy_devices 0 --jax.train_devices 0,1,2,3

    # Run this command at the same time as the command above, to create SAM segmentations. 
    /path/to/conda/env/for/sam/bin/python /path/to/repo/run_sam.py --logdir /preferred/location/for/logs --mode sam --gpus 4,5,6,7


- Sensory Reafferent Benchmark:
    mkdir /preferred/location/for/logs
    mkdir /preferred/location/for/logs/preprocessed_replay
    mkdir /preferred/location/for/logs/postprocessed_replay

    /path/to/conda/env/for/dreamerv3/bin/python /path/to/repo/run_v3.py action_seq_evil_2500_new image_value_gradient image_value_gradient_smooth_with_sam --embed_only_action_adversarial_head True --gradient_weighting_nets policy_function --image_v_grad_interp_value 0.9 --logdir /preferred/location/for/logs --task cheetah_run --jax.policy_devices 0 --jax.train_devices 0,1,2,3

    # Run this command at the same time as the command above, to create SAM segmentations. 
    /path/to/conda/env/for/sam/bin/python /path/to/repo/run_sam.py --logdir /preferred/location/for/logs --mode sam --gpus 4,5,6,7

- Distracting Control Suite Benchmark:
    mkdir /preferred/location/for/logs
    mkdir /preferred/location/for/logs/preprocessed_replay
    mkdir /preferred/location/for/logs/postprocessed_replay

    /path/to/conda/env/for/dreamerv3/bin/python /path/to/repo/run_v3.py --configs natural_evil image_value_gradient image_value_gradient_smooth_with_sam --gradient_weighting_nets policy_function --embed_only_action_adversarial_head True --evil.total_natural_frames 2000 --evil.natural_video_dir="/path/to/kinetics/videos/*.mp4"  --logdir /preferred/location/for/logs --task cheetah_run  --image_v_grad_interp_value 0.9 --jax.policy_devices 0 --jax.train_devices 0,1,2,3

    # Run this command at the same time as the command above, to create SAM segmentations. 
    /path/to/conda/env/for/sam/bin/python /path/to/repo/run_sam.py --logdir /preferred/location/for/logs --mode sam --gpus 4,5,6,7

Example commands for the Denoised MDP experiments in the paper:
- Sensory Reafferent Benchmark:
    /path/to/conda/env/for/dmdp/bin/python /path/to/repo/denoised_mdp-main/main.py output_base_dir=/preferred/location/for/ output_folder=logs env.kind=dmc env.spec=cheetah_run env.evil_level=action_cross_sequence env.action_dims_to_split=[0] env.num_colors_per_cell=2500 env.action_power=4 env.natural_video_dir="/path/to/kinetics/videos/*.mp4" env.total_natural_frames=2000

- Distracting Control Suite Benchmark:
    /path/to/conda/env/for/dmdp/bin/python /path/to/repo/denoised_mdp-main/main.py output_base_dir=/preferred/location/for/ output_folder=logs  env.kind=dmc env.spec=cheetah_run env.natural_video_dir="/path/to/kinetics/videos/*.mp4" env.total_natural_frames=2000 env.evil_level=natural

Example commands for the Dreamer Pro experiments in the paper:
- Sensory Reafferent Benchmark:
    /path/to/conda/env/for/dreamer_pro/bin/python /path/to/repo/dreamer_pro_main/DreamerPro/dreamerv2/train.py --logdir /preferred/location/for/logs --task colorgrid_cheetah_run --configs defaults dmc action_cross_sequence_evil_new

- Distracting Control Suite Benchmark:
    # Note you will need to update the natural video directory in dreamer_pro_main/DreamerPro/dreamerv2/configs.yaml
    /path/to/conda/env/for/dreamer_pro/bin/python /path/to/repo/dreamer_pro_main/DreamerPro/dreamerv2/train.py --logdir /preferred/location/for/logs --task colorgrid_cheetah_run --configs defaults dmc natural_evil

Example commands for the DrQv2 experiments in the paper:
- Sensory Reafferent Benchmark:
/path/to/conda/env/for/drqv2/bin/python /path/to/repo/drqv2/train.py task=cheetah_run background=action_seq_evil_2500_new hydra.run.dir=/preferred/location/for/logs background.natural_video_dir="/path/to/kinetics/videos/*.mp4"

- Distracting Control Suite Benchmark:
/path/to/conda/env/for/drqv2/bin/python /path/to/repo/drqv2/train.py task=cheetah_run background=natural hydra.run.dir=/preferred/location/for/logs background.natural_video_dir="/path/to/kinetics/videos/*.mp4"

Example commands for the Task Informed Abstraction experiments in the paper:
- Sensory Reafferent Benchmark:
/path/to/conda/env/for/tia/bin/python /path/to/repo/run.py --method tia --task dmc_cheetah_run --configs dmc --action_dims_to_split 0 --num_cells_per_dim 16 --num_colors_per_cell 2500 --evil_level action_cross_sequence --action_power 4 --logdir /preferred/location/for/logs

-Distracting Control Suite Benchmark:
/path/to/conda/env/for/tia/bin/python /path/to/repo/run.py --method tia --task dmc_cheetah_run --configs dmc --action_dims_to_split 0 1 2 3 --num_cells_per_dim 16 --num_colors_per_cell 12500 --evil_level natural --total_natural_frames 2000 --natural_video_dir "/path/to/kinetics/videos/*.mp4" --action_power -1 --action_splits 5 10 10 25 --logdir /preferred/location/for/logs