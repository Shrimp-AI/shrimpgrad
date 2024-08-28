import os
import shutil
import urllib.request
import gzip
from shrimpgrad import Tensor, dtypes

url_training = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
url_test = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
url_training_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
url_test_labels = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

_tmp_dir = "/tmp/mnist"
def fetch(url):
  if not os.path.exists(_tmp_dir): os.makedirs(_tmp_dir)
  if not os.path.exists(os.path.join(_tmp_dir, os.path.basename(url))):
    with urllib.request.urlopen(url) as r:
      filename = os.path.join(_tmp_dir, os.path.basename(url))
      with open(filename, 'wb') as f: shutil.copyfileobj(r, f)
    return filename
  print(f"Using cached file {os.path.join(_tmp_dir, os.path.basename(url))}")
  return os.path.join(_tmp_dir, os.path.basename(url))

def mnist():
  def load_data(url, offset): return gzip.open(fetch(url)).read()[offset:]
  return (
    Tensor.frombytes((60000, 1, 28, 28), load_data(url_training, 16), dtypes.uint8),
    Tensor.frombytes((10000,1, 28, 28), load_data(url_test, 16), dtypes.uint8),
    Tensor.frombytes((60000,), load_data(url_training_labels, 8), dtypes.uint8),
    Tensor.frombytes((10000,), load_data(url_test_labels, 8), dtypes.uint8)
  )