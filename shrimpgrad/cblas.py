import ctypes
from ctypes.util import find_library

libcblas = ctypes.CDLL(find_library('cblas'))

# Define the types for the function arguments and return value
libcblas.cblas_sgemm.restype = None
libcblas.cblas_sgemm.argtypes = [
    ctypes.c_int,  # order
    ctypes.c_int,  # transa
    ctypes.c_int,  # transb
    ctypes.c_int,  # m
    ctypes.c_int,  # n
    ctypes.c_int,  # k
    ctypes.c_float,  # alpha
    ctypes.POINTER(ctypes.c_float),  # A
    ctypes.c_int,  # lda
    ctypes.POINTER(ctypes.c_float),  # B
    ctypes.c_int,  # ldb
    ctypes.c_float,  # beta
    ctypes.POINTER(ctypes.c_float),  # C
    ctypes.c_int,  # ldc
]

def sgemm(a, b):
  # Allocate memory for result matrix
  m, k  = a.shape
  k, n = b.shape
  result = (ctypes.c_float * (m * n))()
  # Call cblas_sgemm function
  # If you are using row-major representation then the 
  # number of "columns" will be leading dimension and
  # vice versa in column-major representation number of "rows".
  libcblas.cblas_sgemm(
      ctypes.c_int(101),  # CblasRowMajor
      ctypes.c_int(111),  # CblasNoTrans
      ctypes.c_int(111),  # CblasNoTrans
      ctypes.c_int(m),
      ctypes.c_int(n),
      ctypes.c_int(k),
      ctypes.c_float(1.0),
      (ctypes.c_float * a.size)(*a.data),
      ctypes.c_int(k),
      (ctypes.c_float * b.size)(*b.data),
      ctypes.c_int(n),
      ctypes.c_float(0.0),
      result,
      ctypes.c_int(n)
  )
  return result