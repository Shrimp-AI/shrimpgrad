import time
import types 

def timefunc(fn, *args, **kwargs):
  def fncomposite(*args, **kwargs):
    s = time.monotonic()
    rt = fn(*args, **kwargs)
    e = time.monotonic() - s
    print(f'runtime {fn.__name__} {args[1]} {len(args[2]) if len(args) > 2 else None}={e*1000}ms')
    return rt
  return fncomposite

class Profile(type):
  def __new__(cls, name, bases, attr):
    for name, value in attr.items():
      if type(value) is types.FunctionType or type(value) is types.MethodType: attr[name] = timefunc(value)
    return super(Profile, cls).__new__(cls, name, bases, attr)