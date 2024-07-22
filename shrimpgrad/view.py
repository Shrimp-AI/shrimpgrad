from __future__ import annotations
import functools
import itertools
import operator
from typing import List, Optional, Tuple
from shrimpgrad.util import prod

def can_merge_axes(shape: Tuple[int,...], strides: Tuple[int,...], start:int, stop:int):
  for axis in range(start, stop-1):
    if strides[axis] != strides[axis+1]*shape[axis+1]: return False
  return True

@functools.lru_cache(maxsize=None)
def normalize_strides(shape: Tuple[int, ...], strides: Tuple[int, ...]):
  # replace the stride value for dimensions of 1 with 0
  return tuple([0 if s == 1 else st for s,st in zip(shape, strides)])

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[int, ...]) -> Tuple[int, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[::-1]
  return normalize_strides(shape, strides)

class ViewTracker:
  def __init__(self, views: List[View]):
    self.views: List[View] = views

  @property
  def view(self): return self.views[-1]
  @property
  def shape(self): return self.view.shape
  @property
  def scalar(self): return self.view.scalar
  @property
  def ndim(self): return self.view.ndim
  @property
  def numel(self) -> int: return self.view.numel
  @property
  def contiguous(self): return self.view.contiguous
  @property
  def strides(self): return self.view.strides

  def reshape(self, new_shape: Tuple[int,...]) -> ViewTracker:
    new_view = self.view.reshape(new_shape)
    return ViewTracker.from_views(self.views + [new_view])

  def expand(self, new_shape: Tuple[int,...]) -> ViewTracker:
    return ViewTracker.from_views(self.views + [self.view.expand(new_shape)])

  def permute(self, order: Tuple[int,...]) -> ViewTracker:
    return ViewTracker.from_views(self.views + [self.view.permute(order)] )

  def pad(self, pad_width: Tuple[Tuple[int,int], ...]) -> ViewTracker:
    return ViewTracker.from_views(self.views + [self.view.pad(pad_width)])

  def shrink(self, arg: Tuple[Tuple[int,int], ...]) -> ViewTracker:
    return ViewTracker.from_views(self.views + [self.view.shrink(arg)])

  @staticmethod
  def from_views(views: List[View]) -> ViewTracker:
    return ViewTracker(views)

  @staticmethod
  def from_shape(shape: Tuple[int,...]) -> ViewTracker:
    return ViewTracker([View(shape)])

  def __repr__(self) -> str:
    return f"<VT views={self.views}>"

def create_view(shape: Tuple[int,...],
                strides: Optional[Tuple[int,...]]=None):
  # standardize 0 in shape
  if 0 in shape: return View(shape, (0,)*len(shape))
  return View(shape, normalize_strides(shape, strides) if strides is not None else strides)

class View:
  """
  The view of a thunk's underlying data buffer.
  
  TODO:
  Something that defines slices within the buffer that have actual backing data
    - After pad and shrink we have virtual expansion/contraction of the dimension and we want
      to keep things zero copy i.e.) on pad don't copy the buffer to a new location and fill in zeros where
      they are needed allocating a bunch of memory for the padding that's unecessary.
      - Instead pretend we have been padded, and at realize codegen if values fall out of the valid range use the padded value
      - In this way we keep the original size but don't have to actually store all the padded intermediate tensors  
    - An offset used for computing loop indices, where values that fall outside of the valid range are defaulted to the pad value
  Symbolic Views for variable length tensors i.e. GPT2
  - A way to define shapes that have variable dimensions i.e. Can range from 0 to 100 shape = (Variable('x', 0, 100))
  
  """
  def __init__(self, shape: Tuple[int,...],
               strides: Optional[Tuple[int,...]]=None,
               mask: Optional[Tuple[Tuple[int, int], ...]]=None,
               offset: int=0):
    self.shape = shape
    self.mask: Optional[Tuple[Tuple[int, int], ...]] = mask 
    self.offset: int = offset 
    self.strides: Tuple[int,...] = strides if strides is not None else strides_for_shape(shape)

  @property
  def contiguous(self) -> bool: return self.strides == strides_for_shape(self.shape)
  @property
  def scalar(self): return self.ndim == 0
  @property
  def numel(self) -> int: return prod(self.shape)
  @property
  def ndim(self): return len(self.shape)

  def reshape(self, new_shape: Tuple[int,...]) -> View:
    if len(self.shape):
      assert prod(new_shape) == self.numel, f'shape \'{new_shape}\' is invalid for input of size {self.numel} of shape {self.shape}'
      # Fast path (new strides are easy to compute)
      if self.contiguous: return create_view(new_shape)
      # Slow path (reconstruct the new strides without copying)
      new_strides = self._attempt_no_copy_reshape(new_shape)
      assert new_strides is not None, "failed to reshape"
      return create_view(new_shape, tuple(new_strides))
    return create_view(new_shape)

  def _attempt_no_copy_reshape(self, new_shape):
    # Remove ones from the old shape
    newnd = len(new_shape)
    newdims = new_shape
    olddims = [s for s in self.shape if s != 1]
    oldnd = len(olddims)
    oldstrides = [st for i,st in enumerate(self.strides) if self.shape[i] != 1]
    oi, oj, ni, nj = 0,1,0,1

    newstrides = [0]*len(newdims)
    while ni < newnd and oi < oldnd:
      np = newdims[ni]
      op = olddims[oi]
      while op != np:
        if np < op:
          np *= new_shape[nj]
          nj+=1
        else:
          op *= olddims[oj]
          oj+=1

      if not can_merge_axes(tuple(olddims), tuple(oldstrides), oi, oj):
        return None

      newstrides[nj-1] = oldstrides[oj-1]
      for nk in range(nj-1, ni, -1):
        newstrides[nk-1] = newstrides[nk]*newdims[nk]
      ni = nj
      nj+=1
      oi = oj
      oj+=1

    # Add strides of 0 for trailing 1s
    for nk in range(ni, newnd):
      newstrides[nk] = 0
    return newstrides

  def permute(self, order: Tuple[int,...]) -> View:
    new_shape = tuple([self.shape[i] for i in order])
    new_strides = tuple([self.strides[i] for i in order])
    return create_view(new_shape, tuple(new_strides))

  def expand(self, shape: Tuple[int,...]) -> View:
    assert all(((s0 == s1) or (s0 == 1) for s0,s1 in zip(self.shape, shape))), f'invalid expand from {self.shape} to {shape}'
    strd = list(self.strides)
    for i, (si, so) in enumerate(zip(self.shape, shape)):
      if si != so: strd[i] = 0
    return create_view(shape, tuple(strd))

  def pad(self, pad_width: Tuple[Tuple[int,int],...]) -> View:
    assert all(s >= 0 and e >= 0 for s,e in pad_width), "pad_width must all be >= 0"
    assert len(pad_width) == self.ndim, f'pad_width length must equal view ndim: {len(pad_width) != self.ndim}'
    # No padding needed
    if all(s == 0 and e == 0 for s,e in pad_width): return self
    new_shape = list(self.shape)
    for i, (pad_start, pad_end) in enumerate(pad_width):
      new_shape[i] += pad_start + pad_end
    return create_view(tuple(new_shape), self.strides)

  def shrink(self, arg: Tuple[Tuple[int, int],...]) -> View:
    assert all(0<=start<=stop<=shape for ((start,stop), shape) in zip(arg, self.shape)), 'invalid shrink slices'
    new_shape = tuple([stop - start for start, stop in arg])
    return create_view(new_shape)

  @staticmethod
  def from_view(view: View): return create_view(view.shape, view.strides)

  def __repr__(self): return f'<View shape={self.shape} strides={self.strides} contig={self.contiguous}>'