import concurrent.futures
import functools
import os.path
import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np
import tqdm
from watchdog import events
from watchdog import observers

from . import chunk as chunklib
from dreamerv3.embodied.core import path as pathlib
from . import saver


class Generic:

  def __init__(
      self, length, capacity, remover, sampler, limiter, directory,
      overlap=None, online=False, chunks=1024):
    assert capacity is None or 1 <= capacity
    self.length = length
    self.capacity = capacity
    self.remover = remover
    self.sampler = sampler
    self.limiter = limiter
    self.stride = 1 if overlap is None else length - overlap
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.counters = defaultdict(int)
    self.table = {}
    self.online = online
    if self.online:
      self.online_queue = deque()
      self.online_stride = length
      self.online_counters = defaultdict(int)
    self.saver = directory and saver.Saver(directory, chunks)
    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'insert_wait_dur': 0,
        'insert_wait_count': 0,
    }
    self.load()

  def __len__(self):
    return len(self.table)

  @property
  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    stats = {
        'size': len(self),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
        'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
        'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
        'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def add(self, step, worker=0, load=False):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    stream = self.streams[worker]
    stream.append(step)
    self.saver and self.saver.add(step, worker)
    self.counters[worker] += 1
    if self.online:
      self.online_counters[worker] += 1
      if len(stream) >= self.length and (
          self.online_counters[worker] >= self.online_stride):
        self.online_queue.append(tuple(stream))
        self.online_counters[worker] = 0
    if len(stream) < self.length or self.counters[worker] < self.stride:
      return
    self.counters[worker] = 0
    key = embodied.uuid()
    seq = tuple(stream)
    if load:
      assert self.limiter.want_load()[0]
    else:
      dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
      self.metrics['inserts'] += 1
      self.metrics['insert_wait_dur'] += dur
      self.metrics['insert_wait_count'] += int(dur > 0)
    self.table[key] = seq
    self.remover[key] = seq
    self.sampler[key] = seq
    while self.capacity and len(self) > self.capacity:
      self._remove(self.remover())

  def _sample(self):
    dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)
    if self.online:
      try:
        seq = self.online_queue.popleft()
      except IndexError:
        seq = self.table[self.sampler()]
    else:
      seq = self.table[self.sampler()]
    seq = {k: [step[k] for step in seq] for k in seq[0]}
    seq = {k: embodied.convert(v) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _remove(self, key):
    wait(self.limiter.want_remove, 'Replay remove is waiting')
    del self.table[key]
    del self.remover[key]
    del self.sampler[key]

  def dataset(self):
    while True:
      yield self._sample()

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=False):
    if not self.saver:
      return
    self.saver.save(wait)
    # return {
    #     'saver': self.saver.save(wait),
    #     # 'remover': self.remover.save(wait),
    #     # 'sampler': self.sampler.save(wait),
    #     # 'limiter': self.limiter.save(wait),
    # }

  def load(self, data=None):
    if not self.saver:
      return
    workers = set()
    for step, worker in self.saver.load(self.capacity, self.length):
      workers.add(worker)
      self.add(step, worker, load=True)
    for worker in workers:
      del self.streams[worker]
      del self.counters[worker]
    # self.remover.load(data['remover'])
    # self.sampler.load(data['sampler'])
    # self.limiter.load(data['limiter'])


def wait(predicate, message, sleep=0.001, notify=1.0):
  start = time.time()
  notified = False
  while True:
    allowed, detail = predicate()
    duration = time.time() - start
    if allowed:
      return duration
    if not notified and duration >= notify:
      print(f'{message} ({detail})')
      notified = True
    time.sleep(sleep)


class GenericProcessed:
  def __init__(self, length, capacity, remover, sampler, limiter,
               preprocess_directory, postprocess_directory,
               max_chunks_behind,
               overlap=None, online=False, chunks=256):
    assert capacity is None or 1 <= capacity
    self.length = length
    self.capacity = capacity
    self.remover = remover
    self.sampler = sampler
    self.limiter = limiter
    if overlap is not None:
      raise NotImplementedError(
          'Overlap not supported due to extra burden for postprocessor.')
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.table = {}
    if online:
      raise NotImplementedError('Online not yet supported')
    assert preprocess_directory is not None
    self.saver = preprocess_directory and saver.Saver(
        preprocess_directory, chunks)
    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'postprocess_insert_wait_dur': 0,
        'postprocess_insert_wait_count': 0,
        'postprocess_inserts': 0,
    }
    assert postprocess_directory is not None
    self.postprocess_directory = postprocess_directory
    self.postprocess_handler = PostprocessedFileHandler(
        self._add_step_from_post)
    self.postprocess_observer = observers.Observer()
    self.postprocess_observer.schedule(
        self.postprocess_handler, path=str(self.postprocess_directory),
        recursive=False)
    self.postprocess_observer.start()
    self.first_add_call = True
    self.total_preprocessed = 0
    self.total_postprocesed = 0
    self.max_chunks_behind = max_chunks_behind
    self.postprocess_handler.initial_load(
      self.postprocess_directory, self.capacity, self.length)

  def __len__(self):
    return len(self.table)

  def len_after_postprocessing_completes(self):
    return max(self.total_preprocessed - self.length + 1, 0)

  def process_pending_not_too_large(self):
    if (
        (self.total_preprocessed - self.total_postprocesed)
        // self.length <= self.max_chunks_behind):
      return True, 'Not too far behind'
    else:
      return False, 'Too far behind'

  @property
  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    stats = {
      'size': len(self),
      'inserts': m['inserts'],
      'postprocess_inserts': m['postprocess_inserts'],
      'samples': m['samples'],
      'postprocess_insert_wait_avg': ratio(
          m['postprocess_insert_wait_dur'], m['postprocess_inserts']),
      'postprocess_insert_wait_frac': ratio(
          m['postprocess_insert_wait_count'], m['postprocess_inserts']),
      'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
      'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
      'total_preprocessed': self.total_preprocessed,
      'total_postprocessed': self.total_postprocesed
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  def add(self, step, worker=0):
    if worker != 0:
      raise NotImplementedError('Multiple workers are currently not supported')

    self.metrics['inserts'] += 1
    self.total_preprocessed += 1

    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))

    self.saver and self.saver.add(step, worker)
    if self.first_add_call:
      self.postprocess_handler.next_chunk_uuid = self.saver.buffers[0].uuid
      self.first_add_call = False

  def _sample(self):
    dur = wait(self.limiter.want_sample, 'Replay sample is waiting')
    wait(self.process_pending_not_too_large,
         'Waiting for more examples to be processed')
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur > 0)

    seq = self.table[self.sampler()]
    seq = {
        k: [
            np.unpackbits(step[k], axis=-1).astype(np.bool_) if k == 'masks' else step[k]
            for step in seq
        ]
        for k in seq[0]
    }
    seq = {k: embodied.convert(v) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _remove(self, key):
    wait(self.limiter.want_remove, 'Replay remove is waiting')
    del self.table[key]
    del self.remover[key]
    del self.sampler[key]

  def dataset(self):
    while True:
      yield self._sample()

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=False):
    self.saver.save(wait)

  def load(self, data=None):
    print('load called')

  def _add_step_from_post(self, step, worker, load=False):
    stream = self.streams[worker]
    if 'masks' in step:
      assert step['masks'].shape[-1] % 8 == 0
      step['masks'] = np.packbits(step['masks'], axis=-1)
    stream.append(step)

    if len(stream) < self.length:
      return

    key = embodied.uuid()
    seq = tuple(stream)
    if load:
      assert self.limiter.want_load()[0]
      self.total_preprocessed += 1
    else:
      dur = wait(self.limiter.want_insert, 'Replay insert is waiting')
      self.metrics['postprocess_inserts'] += 1
      self.metrics['postprocess_insert_wait_dur'] += dur
      self.metrics['postprocess_insert_wait_count'] += int(dur > 0)
    self.total_postprocesed += 1
    self.table[key] = seq
    self.remover[key] = seq
    self.sampler[key] = seq

    while self.capacity and len(self) > self.capacity:
      self._remove(self.remover())


class PostprocessedFileHandler(events.FileSystemEventHandler):
  def __init__(self, add_chunk_callable):
    self.add_chunk_callable = add_chunk_callable
    self.next_chunk_uuid = None
    self.loaded_chunks = {}

  def initial_load(self, directory, capacity, length):
    filenames = chunklib.Chunk.scan(directory, capacity, length - 1)

    if not filenames:
      return

    with concurrent.futures.ThreadPoolExecutor(32) as load_executor:
      chunks = list(load_executor.map(
          functools.partial(chunklib.Chunk.load, lazy_load=True), filenames))
      streamids = {}
      for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
        if chunk.successor not in streamids:
          streamids[chunk.uuid] = int(embodied.uuid())
        else:
          streamids[chunk.uuid] = streamids[chunk.successor]

      for i, chunk in tqdm.tqdm(
          enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        stream = streamids[chunk.uuid]
        self.process_chunk(chunk, stream, load=True)
        # Free memory early to not require twice the replay capacity.
        chunks[i] = None
        del chunk

  def on_moved(self, event):
    assert isinstance(event, events.FileMovedEvent)
    assert self.next_chunk_uuid is not None, (
        'Chunk UUID should have been set by now.')
    chunk = chunklib.Chunk.load(pathlib.Path(event.dest_path))
    self.loaded_chunks[chunk.uuid] = chunk
    self.process_chunks()

  def process_chunks(self):
    while self.next_chunk_uuid in self.loaded_chunks:
      chunk = self.loaded_chunks.pop(self.next_chunk_uuid)
      if chunk.successor == str(embodied.uuid(0)):
        continue  # These tiny chunks are created when saver calls.
                  # They can be thrown away.
      self.next_chunk_uuid = chunk.successor
      self.process_chunk(chunk, 0)

  def process_chunk(self, chunk, stream, load=False):
    for index in range(chunk.length):
      step = {k: v[index] for k, v in chunk.data.items()}
      self.add_chunk_callable(step, stream, load=load)
