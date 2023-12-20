// A tensor will be stored in row-major order on the heap.
// We allocate dtype * prod(shape) space on the heap.
// Initial dtype will be float64.

const std = @import("std");
const debug = std.debug;
const assert = debug.assert;
const testing = std.testing;
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const ArrayList = std.ArrayList;

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        shape: []usize,
        size: usize,
        strides: Strides(),
        allocator: Allocator,
        items: Slice,
        capacity: usize,

        pub const Slice = []align(@alignOf(T)) T;

        pub fn init(allocator: Allocator, shape: []usize) Allocator.Error!Self {
            var size: usize = 1;
            for (shape) |dim_size| {
                size *= dim_size;
            }
            const strides = try Strides().init(@constCast(&shape).*, size, allocator);
            return Self{
                .shape = shape,
                .size = size,
                .allocator = allocator,
                .strides = strides,
                .items = &[_]T{},
                .capacity = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.strides.deinit();
            self.allocator.free(self.items.ptr[0..self.capacity]);
        }

        pub fn allocate(self: *Self) Allocator.Error!void {
            const new_memory = try self.allocator.alignedAlloc(T, @alignOf(T), self.size);
            self.items.ptr = new_memory.ptr;
            self.capacity = new_memory.len;
        }

        // Ctors
        pub fn zeros(allocator: Allocator, shape: []usize) Allocator.Error!Self {
            var tensor = try init(allocator, shape);
            try tensor.allocate();
            @memset(tensor.items.ptr[0..tensor.capacity], 0);
            return tensor;
        }

        const TensorError = error{
            IndexIncomplete,
            OutOfBounds,
        };

        pub fn get(self: *Self, index: []usize) TensorError!T {
            if (index.len != self.shape.len) {
                return TensorError.IndexIncomplete;
            }
            var loc: usize = 0;
            for (index, 0..) |idx, i| {
                loc += idx * self.strides.items[i];
            }

            if (loc >= self.capacity) {
                return TensorError.OutOfBounds;
            }
            return self.items.ptr[loc];
        }
    };
}

pub fn Strides() type {
    return struct {
        const Self = @This();

        strides: std.ArrayList(usize),
        items: []usize,
        pub fn init(shape: []usize, size: usize, allocator: Allocator) Allocator.Error!Self {
            var strides = std.ArrayList(usize).init(allocator);

            var out: usize = 1;
            for (0..shape.len) |i| {
                out *= shape[i];
                const stride = size / out;
                try strides.append(stride);
            }

            return Self{
                .items = strides.items,
                .strides = strides,
            };
        }

        pub fn deinit(self: *Self) void {
            self.strides.deinit();
        }

        pub fn get(self: *Self, i: usize) usize {
            return self.strides.items[i];
        }
    };
}

const CTensor = extern struct {
    data: [*]f32,
    size: usize,
    shape_len: usize,
    shape: [*]usize,
    strides: [*]usize,
};

const FloatTensor = Tensor(f32);

export fn zeros(shape: [*]usize, shape_len: usize) CTensor {
    const allocator = std.heap.page_allocator;
    const tensor = FloatTensor.zeros(allocator, shape[0..shape_len]) catch unreachable;

    return CTensor{
        .data = tensor.items.ptr,
        .size = tensor.size,
        .shape_len = shape_len,
        .shape = shape,
        .strides = tensor.strides.items.ptr,
    };
}

export fn tensorDeinit(tensor: *CTensor) void {
    const allocator = std.heap.page_allocator;
    allocator.free(tensor.data[0..tensor.size]);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var shape: [3]usize = .{ 4, 5, 6 };
    var tensor = try Tensor(f64).zeros(gpa.allocator(), &shape);
    defer tensor.deinit();
    const strides = tensor.strides.items;
    var index = [3]usize{ 1, 1, 1 };
    debug.print("strides={},{},{}, capacity={} tensor[0]={}", .{ strides[0], strides[1], strides[2], tensor.capacity, try tensor.get(&index) });
}
