import abc
import argparse
import concurrent.futures
import glob
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
import traceback
import warnings

import numpy as np
from PIL import Image
from sam2 import automatic_mask_generator
from sam2 import build_sam
import torch
from transformers import pipeline
from watchdog import events
from watchdog import observers


_MAX_MASKS = 300


class FileProcessor(events.FileSystemEventHandler, abc.ABC):
  _NUM_THREADS = 4  # Expected number of GPUs for the real thing.

  def __init__(self, preprocessed_directory, postprocessed_directory):
    self.preprocessed_directory = preprocessed_directory
    self.postprocessed_directory = postprocessed_directory
    self.executor = concurrent.futures.ThreadPoolExecutor(self._NUM_THREADS)

  def _get_all_files_in_dir(self, dir):
    return [
      os.path.basename(path) for path in glob.glob(os.path.join(dir, '*.npz'))]

  @abc.abstractmethod
  def _process_file(self, src, dest):
    """
    Abstract method to process a file.
    This needs to be implemented in the child class.
    """
    pass

  def initial_load(self):
    preprocessed_files = set(
      self._get_all_files_in_dir(self.preprocessed_directory))
    postprocessed_files = set(
      self._get_all_files_in_dir(self.postprocessed_directory))
    unprocessed_files = preprocessed_files - postprocessed_files
    tasks = []
    for file in unprocessed_files:
      src_path = os.path.join(self.preprocessed_directory, file)
      dst_path = os.path.join(self.postprocessed_directory, file)
      print(f'enqueuing {file}')
      tasks.append(
        self.executor.submit(self._process_file, src_path, dst_path))

    for task in tasks:
      task.result()

  def on_moved(self, event):
    assert isinstance(event, events.FileMovedEvent)
    file_name = os.path.basename(event.dest_path)
    print(f'enqueuing {file_name}')
    self.executor.submit(
      self._process_file, event.dest_path,
      os.path.join(self.postprocessed_directory, file_name))


class SAMEmulatorFileProcessor(FileProcessor):
  _DELAY_MIN = 15  # seconds
  _DELAY_MAX = 30  # seconds

  def _process_file(self, src, dest):
    time.sleep(random.uniform(self._DELAY_MIN, self._DELAY_MAX))
    shutil.copy(src, dest + '.tmp')
    shutil.move(dest + '.tmp', dest)


def _pad_masks(masks, max_masks):
  padded_masks = np.zeros((max_masks, *masks[0].shape), dtype=masks[0].dtype)
  padded_masks[:min(len(masks), max_masks)] = masks[:max_masks]
  return padded_masks


def _setup_pipeline(gpu_indices: mp.Queue, src_dest_queue: mp.Queue):
  # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices.get()
  gpu = gpu_indices.get()
  generator = pipeline(
      'mask-generation', model='facebook/sam-vit-base', device=gpu)

  while True:
    src, dest = src_dest_queue.get()

    # Suppressing specific warning
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      chunk = np.load(src)

      # Generate masks for all images in a batch
      images = [Image.fromarray(image) for image in chunk['image']]
      try:
        batch_outputs = generator(
            images, pred_iou_thresh=.75, stability_score_thresh=.75,
            points_per_crop=16)
      except Exception:
        logging.error(f"Can't process {src}: {traceback.format_exc()}")
        continue

      # Pad masks and prepare for saving
      masks_count = []
      padded_masks_list = []
      for output in batch_outputs:
        masks = np.array(output['masks'])
        masks_count.append(min(len(masks), _MAX_MASKS))
        padded_masks = _pad_masks(masks, _MAX_MASKS)
        padded_masks_list.append(padded_masks)

      # Stack and save the masks
      all_masks = np.stack(padded_masks_list)
      masks_count_array = np.array(masks_count)
      np.savez_compressed(dest.replace('.npz', 'tmp.npz'), **chunk, masks=all_masks,
                          masks_count=masks_count_array)
      shutil.move(dest.replace('.npz', 'tmp.npz'), dest)


class SamFileProcessor(FileProcessor):
  def __init__(self, gpu_indices, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._gpu_queue = mp.Queue()
    for idx in gpu_indices:
      self._gpu_queue.put(idx)
    self._src_dest_queue = mp.Queue()
    self.processes = [mp.Process(
      target=_setup_pipeline, args=(self._gpu_queue, self._src_dest_queue))
      for _ in gpu_indices
    ]
    for process in self.processes:
      process.start()

  def _process_file(self, src, dest):
    self._src_dest_queue.put((src, dest))


def _setup_sam_2_pipeline(
    gpu_indices: mp.Queue, src_dest_queue: mp.Queue, sam_checkpoint: str,
    sam_model_cfg: str):
  torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
  if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

  gpu = gpu_indices.get()

  sam2 = build_sam.build_sam2(
    sam_model_cfg, sam_checkpoint, device=f'cuda:{gpu}')
  mask_generator = automatic_mask_generator.SAM2AutomaticMaskGenerator(
    sam2, points_per_side=16, pred_iou_thresh=.75, stability_score_thresh=.75,
    output_mode='binary_mask')

  while True:
    src, dest = src_dest_queue.get()
    chunk = np.load(src)

    # Generate masks for all images in a batch
    images = chunk['image']
    try:
      batch_outputs = [mask_generator.generate(image) for image in images]
    except Exception:
      logging.error(f"Can't process {src}: {traceback.format_exc()}")
      continue

    # TODO: Deduplicate the code below with _setup_sam_pipeline.
    # Pad masks and prepare for saving
    masks_count = []
    padded_masks_list = []
    for output in batch_outputs:
      masks_count.append(min(len(output), _MAX_MASKS))
      masks = [record['segmentation'] for record in output]
      padded_masks = _pad_masks(masks, _MAX_MASKS)
      padded_masks_list.append(padded_masks)

    # Stack and save the masks
    all_masks = np.stack(padded_masks_list)
    masks_count_array = np.array(masks_count)
    np.savez_compressed(dest.replace('.npz', 'tmp.npz'), **chunk,
                        masks=all_masks,
                        masks_count=masks_count_array)
    shutil.move(dest.replace('.npz', 'tmp.npz'), dest)


class Sam2FileProcessor(FileProcessor):
  def __init__(self, gpu_indices, sam_checkpoint: str, sam_model_config: str, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._gpu_queue = mp.Queue()
    for idx in gpu_indices:
      self._gpu_queue.put(idx)
    self._src_dest_queue = mp.Queue()
    self.processes = [
      mp.Process(target=_setup_sam_2_pipeline, args=(
        self._gpu_queue, self._src_dest_queue, sam_checkpoint,
        sam_model_config))
      for _ in gpu_indices
    ]
    for process in self.processes:
      process.start()

  def _process_file(self, src, dest):
    self._src_dest_queue.put((src, dest))


def main():
  mp.set_start_method('spawn')
  parser = argparse.ArgumentParser(
    description='Use SAM to create postprocessed version of saved chunks '
                'from DreamerV3 with additional segmentation masks for '
                'the images.')
  parser.add_argument(
    '--logdir', required=True, help='Directory for log files')
  parser.add_argument(
      '--mode', required=True, help='Mode to operate in'
  )
  parser.add_argument(
    '--gpus', help='GPUs to use.', default='0'
  )
  parser.add_argument('--sam2_ckpt', default='/path/to/sam2_ckpt.pt')
  parser.add_argument('--sam2_cfg', default='sam2_hiera_l.yaml')
  args = parser.parse_args()

  # Check if logdir exists
  if not os.path.exists(args.logdir):
    print(f"The specified log directory '{args.logdir}' does not exist.")
    return

  # Create paths for preprocessed and postprocessed replay directories
  preprocessed_replay_path = os.path.join(args.logdir, 'preprocessed_replay')
  postprocessed_replay_path = os.path.join(args.logdir,
                                           'postprocessed_replay')

  if not os.path.exists(preprocessed_replay_path):
    print(f'Specified preprocessed directory does not exist.')
    return

  if not os.path.exists(postprocessed_replay_path):
    print(f'Specified postprocessed directory does not exist.')
    return

  if args.mode == 'emulate':
    sam_handler = SAMEmulatorFileProcessor(
        preprocessed_replay_path, postprocessed_replay_path)
  elif args.mode == 'sam':
    sam_handler = SamFileProcessor(
        [int(i) for i in args.gpus.split(',')],
        preprocessed_replay_path, postprocessed_replay_path)
  elif args.mode == 'sam_2':
    sam_handler = Sam2FileProcessor(
        [int(i) for i in args.gpus.split(',')],
        args.sam2_ckpt, args.sam2_cfg,
        preprocessed_replay_path, postprocessed_replay_path)
  else:
    raise ValueError('Mode must be either emulate or sam')
  sam_handler.initial_load()
  observer = observers.Observer()
  observer.schedule(
      sam_handler, path=preprocessed_replay_path, recursive=False)
  observer.start()
  try:
    while True:
      time.sleep(1)
  finally:
    observer.stop()
    observer.join()


if __name__ == '__main__':
  main()
