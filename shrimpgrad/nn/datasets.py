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

def mnist_loader(batch_size:int):
  train_images, test_images, train_labels, test_labels = mnist()
  train_images = train_images.shrink(((0,batch_size), None, None, None))
  test_images = test_images.shrink(((0,batch_size), None, None, None))
  train_labels = train_labels.shrink(((0,batch_size),))
  test_labels = test_labels.shrink(((0,batch_size),))
  return train_images, train_labels, test_images, test_labels