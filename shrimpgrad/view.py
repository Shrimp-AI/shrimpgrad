from __future__ import annotations
from itertools import accumulate
import operator
from typing import List, Tuple
from shrimpgrad.util import prod

def can_merge_axes(shape: Tuple[int,...], strides: Tuple[int,...], start:int, stop:int):
  for axis in range(start, stop-1):
    if strides[axis] != strides[axis+1]*shape[axis+1]: return False
  return True

def normalize_strides(shape: Tuple[int, ...], strides: Tuple[int, ...]):
  # replace the stride value for dimensions of 1 with 0
  return tuple([0 if s == 1 else st for s,st in zip(shape, strides)])

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
  def numel(self): return self.view.numel
  @property
  def contiguous(self): return self.view.contiguous
  @property
  def strides(self): return self.view.strides

  def reshape(self, new_shape: Tuple[int,...]) -> ViewTracker:
    new_view = self.view.reshape(new_shape)
    if self.view.contiguous:
      # if the most recent view is not permuted
      # we can just merge the views
      return ViewTracker.from_views(self.views[0:-1] + [new_view])
    # Must respect the permute
    return ViewTracker.from_views(self.views + [new_view])

  def expand(self, new_shape: Tuple[int,...]) -> ViewTracker:
    return ViewTracker.from_views(self.views[0:-1] + [self.view.expand(new_shape)])

  def permute(self, order: Tuple[int,...]) -> ViewTracker:
    return ViewTracker.from_views(self.views[0:-1] + [self.view.permute(order)] )

  def pad(self, pad_width: Tuple[Tuple[int,int], ...]) -> ViewTracker:
    return ViewTracker.from_views(self.views[0:-1] + [self.view.pad(pad_width)])

  @staticmethod
  def from_views(views: List[View]) -> ViewTracker:
    return ViewTracker(views)

  @staticmethod
  def from_shape(shape: Tuple[int,...]) -> ViewTracker:
    return ViewTracker([View(shape)])

  def __repr__(self) -> str:
    return f"<VT views={self.views}>"


class View:
  """A description of how a thunk's data is interpreted
  """
  def __init__(self, shape: Tuple[int,...], mask=None):
    self.shape = shape
    self._strides = tuple(accumulate(self.shape[-1:0:-1], func=operator.mul, initial=(1 if len(self.shape)else None)))[::-1]
    self.mask = mask

  @property
  def strides(self) -> Tuple[int,...]: return self._strides

  @property
  def contiguous(self) -> bool:
    if not self.shape: return True
    if not self._strides: return True
    return all(self._strides[i] == self.shape[i+1]*self._strides[i+1] for i in range(0, self.ndim-1))

  @property
  def scalar(self): return self.ndim == 0
  @property
  def numel(self): return prod(self.shape)
  @property
  def ndim(self): return len(self.shape)

  def reshape(self, new_shape: Tuple[int,...]) -> View:
    if len(self.shape):
      assert prod(new_shape) == self.numel, f'shape \'{new_shape}\' is invalid for input of size {self.numel} of shape {self.shape}'
      # Fast path (new strides are easy to compute)
      if self.contiguous: return View(new_shape)
      # Slow path (reconstruct the new strides without copying)
      newstrides = self._attempt_no_copy_reshape(new_shape)
      view = View(new_shape)
      view._strides = normalize_strides(new_shape, tuple(newstrides))
      return view
    return View(new_shape)

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

      if not can_merge_axes(olddims, oldstrides, oi, oj):
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
    v = View(new_shape)
    v._strides = new_strides
    return v

  def expand(self, shape: Tuple[int,...]) -> View:
    out = View.from_view(self)
    strd = list(self.strides)
    for i, (si, so) in enumerate(zip(self.shape, shape)):
      if si != so: strd[i] = 0
    out.shape = shape
    out._strides = tuple(strd)
    return out

  def pad(self, pad_width: Tuple[Tuple[int,int],...]):
    assert all(s >= 0 and e >= 0 for s,e in pad_width), "pad_width must all be >= 0"
    assert len(pad_width) == self.ndim, f'pad_width length must equal view ndim: {len(pad_width) != self.ndim}'

    # No padding needed
    if all(s == 0 and e == 0 for s,e in pad_width): return self
    new_shape = list(self.shape)
    mask = [None]*self.ndim
    for i, ((pad_start, pad_end), shp) in enumerate(zip(pad_width, self.shape)):
      new_shape[i] += pad_start + pad_end
      # start index of non-padded values, end value of non-padded values
      mask[i] = (pad_start, shp + pad_start)
    return View(tuple(new_shape), tuple(mask))

  @staticmethod
  def from_view(view: View):
    return View(view.shape)

  def __repr__(self): return f'<View shape={self.shape} strides={self.strides} contig={self.contiguous}>'