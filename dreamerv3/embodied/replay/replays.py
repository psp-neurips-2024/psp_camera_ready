from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )


class UniformProcessed(generic.GenericProcessed):
  def __init__(
        self, length, capacity=None, preprocess_directory=None,
        postprocess_directory=None, max_chunks_behind=None, online=False,
        chunks=256,  # Reduce default chunk size as the chunks serve as shards
                    # for the postprocessor.
        min_size=4,  # Increase min size to 4 to give the postprocessor some
                     # time to catch up with the first experience gathering.
        samples_per_insert=None,
        tolerance=1e4,
        seed=0
  ):
    if samples_per_insert:
        limiter = limiters.SamplesPerInsert(
            samples_per_insert, tolerance, min_size)
    else:
        limiter = limiters.MinSize(min_size)
    assert not capacity or min_size <= capacity
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        preprocess_directory=preprocess_directory,
        postprocess_directory=postprocess_directory,
        max_chunks_behind=max_chunks_behind,
        online=online,
        chunks=chunks,
    )
