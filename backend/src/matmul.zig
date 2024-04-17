const std = @import("std");

const c = @cImport({
    @cInclude("cblas.h");
});

const TensorError = error{
    IndexError,
};

pub fn matmul(a: [][]f64, b: [][]f64) ![][]f64 {
    if (a[0].len != b.len) {
        return error.IndexError;
    }

    const result: [][]f64 = std.mem.zeroes([][0]f64, a.len);
    const aPtr: [*]f64 = a.*;
    const bPtr: [*]f64 = b.*;
    const resultPtr: [*]f64 = result.*;

    _ = c.cblas_dgemm(c.CblasRowMajor, c.CblasNoTrans, c.CblasNoTrans, a.len, b[0].len, a[0].len, 1.0, aPtr, a[0].len, bPtr, b[0].len, 0.0, resultPtr, b[0].len);

    return result;
}
