# Example usage
from shrimpgrad.shrimp_ffi import Z, shape

ctensor = Z.zeros(shape((3, 3)), 2)

# Access tensor data...
for i in range(ctensor.size):
  print(ctensor.data[i])

# When done, free the resources
Z.deinit(ctensor)
