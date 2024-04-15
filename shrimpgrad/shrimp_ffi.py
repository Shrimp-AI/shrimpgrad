from cffi import FFI
import pkg_resources

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

dylib_path = pkg_resources.resource_filename(
  'shrimpgrad', 'lib/libshrimpgrad.0.1.0.dylib')

Z = ffi.dlopen(dylib_path)


def shape(s):
  shape = ffi.new("size_t[]", s)
  return shape
