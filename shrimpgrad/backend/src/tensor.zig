// A tensor will be stored in row-major order on the heap.
// We allocate dtype * prod(shape) space on the heap.
// Initial dtype will be float64.

const std = @import("std");

pub fn Tensor(T: type) type {
    _ = T;
}
