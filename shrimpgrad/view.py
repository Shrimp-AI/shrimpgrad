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
                strides: Optional[Tuple[int,...]]=None,
                mask: Optional[Tuple[Tuple[int,int],...]]=None,
                offset: int=0):
  # standardize 0 in shape
  if 0 in shape: return View(shape, (0,)*len(shape))
  return View(shape, normalize_strides(shape, strides) if strides is not None else strides, mask=mask, offset=offset)

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
    new_mask = None
    if self.mask is not None:
      new_mask = _reshape_mask(self.mask, self.shape, new_shape)
    if len(self.shape):
      assert prod(new_shape) == self.numel, f'shape \'{new_shape}\' is invalid for input of size {self.numel} of shape {self.shape}'
      # Fast path (new strides are easy to compute)
      if self.contiguous: return create_view(new_shape)
      # Slow path (reconstruct the new strides without copying)
      new_strides = self._attempt_no_copy_reshape(new_shape)
      if new_strides is None: return create_view(new_shape, mask=new_mask)
      return create_view(new_shape, tuple(new_strides), new_mask)
    return create_view(new_shape, mask=new_mask)

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
    offset = sum([-p[0]*st for p,st in zip(pad_width, self.strides)]) 
    new_shape = tuple([s + ps + pe for s, (ps, pe) in zip(self.shape, pad_width)])
    mask = tuple([(p,p+s) for ((p,_),s) in zip(pad_width, self.shape)])
    return create_view(new_shape, self.strides, mask, offset)

  def shrink(self, arg: Tuple[Tuple[int, int],...]) -> View:
    assert all(0<=s<=e<=sh for ((s,e),sh) in zip(arg, self.shape)), 'invalid shrink slices'
    nmsk = tuple([(start := nms if nms < ms else 0,
                  nme if nme < me else start+(me-ms)) 
                for ((ms,me), (nms,nme)) in zip(self.mask, arg)]) \
                  if self.mask is not None else None
    return create_view(tuple([e-s for s,e in arg]), mask=nmsk)

  @staticmethod
  def from_view(view: View): return create_view(view.shape, view.strides)

  def __repr__(self): return f'<View shape={self.shape} strides={self.strides} contig={self.contiguous} mask={self.mask} offset={self.offset}>'

# TODO: Figure this out more clearly so we can rewrite this (this is the tinygrad implementation)
def _reshape_mask(_mask, old_shape, new_shape):
  if _mask is None: return tuple((0, s) for s in new_shape)
  if any(not isinstance(m[0], int) or not isinstance(m[1], int) for m in _mask): return None
  if any(m[1] - m[0] < 1 for m in _mask): return ((0, 0),) * len(new_shape)  # zero mask

  new_mask = []
  # _mask is all int here
  r_masks, r_shape, r_new_shape = reversed(_mask), reversed(old_shape), reversed(new_shape)
  curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))

  while len(new_mask) < len(new_shape):
    (l, r), next_stride = mask, new_dim * curr_stride
    print(f"{l = } {r = }  {next_stride = } {old_dim = } {new_dim = } {mask = }")

    if old_dim >= next_stride: # need to split mask.
      print("Split mask")
      if old_dim == next_stride: # simply copy the mask and get next batch for merging
        print("Copy")
        new_mask.append((l // curr_stride, (r - 1) // curr_stride + 1))
        curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))

      else: # mask can only be splitted if reshape doesn't cut across the mask.
        print("Check cut across")
        print(f"{l % next_stride = } {r % next_stride = } {l // next_stride = }") 

        if (((l % next_stride != 0 or r % next_stride != 0) and l // next_stride != (r - 1) // next_stride)
            or old_dim % next_stride != 0): 
          print("CUTTED")
          return None
        new_mask.append((l % next_stride // curr_stride, (r - 1) % next_stride // curr_stride + 1))
        curr_stride, new_dim = next_stride,  next(r_new_shape, 1) # need to get mask for next dimension

    else:
      next_mask = next(r_masks, (0, 1))
      print(f"No split needed {next_mask = }")
      # combine if the mask can unfold continuously
      if mask != (0, old_dim) and next_mask[1] - next_mask[0] != 1: return None
      mask, old_dim = (next_mask[0] * old_dim + l, (next_mask[1] - 1) * old_dim + r), old_dim * next(r_shape, 1)
    print(f"{new_mask = }")

  for mask in r_masks: # if the old shape has leading 1s, need to make sure their mask is (0,1)
    if mask != (0, 1): return ((0, 0),) * len(new_shape) # invalid mask

  return tuple(reversed(new_mask))