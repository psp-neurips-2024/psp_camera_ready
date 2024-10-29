import io
from datetime import datetime
import shutil

import embodied
import numpy as np


class Chunk:

  def __init__(self, size, successor=None):
    now = datetime.now()
    self.time = now.strftime("%Y%m%dT%H%M%S") + f'F{now.microsecond:06d}'
    self.uuid = str(embodied.uuid())
    self.successor = successor
    self.size = size
    self._data = None
    self.length = 0
    self._filename = None

  def __repr__(self):
    succ = self.successor or str(embodied.uuid(0))
    succ = succ.uuid if isinstance(succ, type(self)) else succ
    return (
        f'Chunk(uuid={self.uuid}, '
        f'succ={succ}, '
        f'len={self.length})')

  def __len__(self):
    return self.length

  def __bool__(self):
    return True

  def append(self, step):
    if not self._data:
      example = {k: embodied.convert(v) for k, v in step.items()}
      self._data = {
          k: np.empty((self.size,) + v.shape, v.dtype)
          for k, v in example.items()}
    for key, value in step.items():
      self._data[key][self.length] = value
    self.length += 1

  def save(self, directory):
    succ = self.successor or str(embodied.uuid(0))
    succ = succ.uuid if isinstance(succ, type(self)) else succ
    filename = f'{self.time}-{self.uuid}-{succ}-{self.length}.npz'
    tmp_filename = embodied.Path(directory) / (filename + '.tmp')
    filename = embodied.Path(directory) / filename
    data = {k: embodied.convert(v) for k, v in self.data.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **data)
      stream.seek(0)
      tmp_filename.write(stream.read(), mode='wb')
    shutil.move(str(tmp_filename), str(filename))
    print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename, lazy_load=False):
    length = int(filename.stem.split('-')[3])
    chunk = cls(length)
    chunk._filename = filename
    chunk.time = filename.stem.split('-')[0]
    chunk.uuid = filename.stem.split('-')[1]
    chunk.successor = filename.stem.split('-')[2]
    chunk.length = length
    if not lazy_load:
      chunk._load_data()
    return chunk

  def _load_data(self):
    with embodied.Path(self._filename).open('rb') as f:
      data = np.load(f)
      self._data = {k: data[k] for k in data.keys()}

  @property
  def data(self):
    if self._data is None:
      self._load_data()

    return self._data

  @classmethod
  def scan(cls, directory, capacity=None, shorten=0):
    directory = embodied.Path(directory)
    filenames, total = [], 0
    for filename in reversed(sorted(directory.glob('*.npz'))):
      if capacity and total >= capacity:
        break
      filenames.append(filename)
      total += max(0, int(filename.stem.split('-')[3]) - shorten)
    return sorted(filenames)
