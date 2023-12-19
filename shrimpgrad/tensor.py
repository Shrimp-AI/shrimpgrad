import numpy as np


class Tensor:
  def __init__(self, shape=(1,), strides=[0], data=[]):
    self.shape = shape
    self.strides = strides
    self.data = data
