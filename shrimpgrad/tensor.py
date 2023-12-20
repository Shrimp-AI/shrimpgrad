from cffi import FFI

ffi = FFI()

ffi.cdef("""
    typedef struct {
        float *data;
        size_t size;
        size_t shape_len;
        size_t *shape;
        size_t *strides;
    } CTensor;

    CTensor zeros(const size_t* shape, size_t shape_len);
    void deinit(CTensor tensor);
""")

C = ffi.dlopen(
  './backend/zig-out/lib/libshrimpgrad.0.1.0.dylib')

# Example usage
shape = ffi.new("size_t[]", [3, 3])
ctensor = C.zeros(shape, 2)

# Access tensor data...
for i in range(ctensor.size):
  print(ctensor.data[i])

# When done, free the resources
C.tensorDeinit(ctensor)
