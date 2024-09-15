from typing import List, Callable
from shrimpgrad import Tensor, nn 
from shrimpgrad.knobs import Knobs
from shrimpgrad.nn.datasets import mnist_loader
import unittest
from shrimpgrad.engine.jit import ShrimpJit
from shrimpgrad.nn import BatchNorm, LayerNorm, get_parameters, optim
import torch
import numpy as np

maxpool2d = lambda x: Tensor.maxpool2d(x, (2,2),(2,2)) 
class ConvNet:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2D(1, 32, 5), Tensor.relu,
      nn.Conv2D(32, 32, 5), Tensor.relu,
      nn.BatchNorm(32), maxpool2d, 
      nn.Conv2D(32, 64, 3), Tensor.relu,
      nn.Conv2D(64, 64, 3), Tensor.relu,
      nn.BatchNorm(64), maxpool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10)]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

def prepare_tensors(shapes, low=-1.5, high=1.5):
  np.random.seed(0)
  np_data = [np.random.uniform(low=low, high=high, size=shp).astype(np.float32) for shp in shapes]
  tts = [torch.tensor(data, requires_grad=True, dtype=torch.float32) for data in np_data]
  sts = [Tensor(shp, data.flatten().tolist() if len(shp) else data.flatten().tolist()[0], requires_grad=True) for shp, data in zip(shapes, np_data)]
  return sts, tts

class ShrimpModel:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
    nn.Linear(2, 50), Tensor.relu,
    nn.Linear(50, 1), Tensor.sigmoid,
    ]
  def __call__(self, x: Tensor):
    return x.sequential(self.layers)

class TorchModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_relu_stack = torch.nn.Sequential(
        torch.nn.Linear(2, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1),
        torch.nn.Sigmoid()
    )

  @torch.jit.ignore
  def inject_weights(self, w, b, w_, b_):
    m = self.linear_relu_stack[0].eval()
    m.weight[:] = w
    m.bias[:] = b
    m = self.linear_relu_stack[2].eval()
    m.weight[:] = w_
    m.bias[:] = b_

  @torch.jit.ignore
  def set_requires_grad(self):
    self.linear_relu_stack[0].weight.requires_grad = True
    self.linear_relu_stack[0].bias.requires_grad = True
    self.linear_relu_stack[2].weight.requires_grad = True
    self.linear_relu_stack[2].bias.requires_grad = True
    self.linear_relu_stack[0].weight.retain_grad()
    self.linear_relu_stack[0].bias.retain_grad()
    self.linear_relu_stack[2].weight.retain_grad()
    self.linear_relu_stack[2].bias.retain_grad()

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits

def dataset():
  from sklearn.datasets import make_moons
  X, y = make_moons(n_samples=100, noise=0.1)
  X = X.astype(float)
  y = y.astype(float).reshape(100)
  return X, y

class TestNN(unittest.TestCase):
  def test_linear(self):
    x = Tensor((2,2), [1.0,2.0,3.0,4.0])
    model = nn.Linear(2,2)
    model.w.requires_grad = True
    model.bias.requires_grad = True
    z = model(x).square().mean()
    z.realize()
    with torch.no_grad():
      torch_model = torch.nn.Linear(2,2).eval()
      torch_model.weight[:] = torch.tensor(model.w.data().copy()).reshape(2,2)
      torch_model.bias[:] = torch.tensor(model.bias.data().copy())
    torch_model.weight.requires_grad = True
    torch_model.weight.retain_grad()
    torch_model.bias.requires_grad = True
    torch_model.bias.retain_grad()
    torch_x = torch.tensor(x.data().copy()).reshape(2,2)
    torch_z = torch_model(torch_x).square().mean()
    np.testing.assert_allclose(np.array(z.data()), torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)
    torch_z.backward()
    z.backward()
    model.w.grad.realize()
    np.testing.assert_allclose(model.w.grad.data(), torch_model.weight.grad.detach().numpy(), atol=1e-6, rtol=1e-3)

  def test_basic_net(self):
    X, y  = dataset()
    y = y.reshape(100)
    shrimp_model = ShrimpModel()
    torch_model = TorchModel()
    w0 = shrimp_model.layers[0].w
    b0 = shrimp_model.layers[0].bias
    w1 = shrimp_model.layers[2].w
    b1 = shrimp_model.layers[2].bias
    w0.requires_grad = True
    b0.requires_grad = True
    w1.requires_grad = True
    b1.requires_grad = True
    sloss = shrimp_model(Tensor.fromlist(X.shape, X.flatten().tolist())).reshape(100).binary_cross_entropy(Tensor.fromlist(y.shape, y.flatten().tolist()))
    sloss.realize()
    tw0 = torch.tensor(w0.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*w0.shape)
    tb0 = torch.tensor(b0.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*b0.shape)
    tw1 = torch.tensor(w1.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*w1.shape)
    tb1 = torch.tensor(b1.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*b1.shape)
    with torch.no_grad():
      torch_model.inject_weights(tw0,tb0, tw1, tb1)
    torch_model.set_requires_grad()
    tout = torch_model(torch.tensor(X, dtype=torch.float32)).reshape(100)
    tloss = torch.nn.functional.binary_cross_entropy(tout, torch.tensor(y, dtype=torch.float32))

    np.testing.assert_allclose(sloss.data(), tloss.detach().numpy(), atol=1e-6, rtol=1e-2)

    sloss.backward()
    tloss.backward()
    np.testing.assert_allclose(w0.grad.numpy(), torch_model.linear_relu_stack[0].weight.grad.detach().numpy(), atol=1e-4, rtol=1e-3)

  def test_basic_net_training(self):
    X, y  = dataset()
    y = y.reshape(100)
    # Build the models, realize shrimp model to transfer the weights to the torch model
    shrimp_model = ShrimpModel()
    torch_model = TorchModel()

    w0 = shrimp_model.layers[0].w
    b0 = shrimp_model.layers[0].bias
    w1 = shrimp_model.layers[2].w
    b1 = shrimp_model.layers[2].bias

    sloss = shrimp_model(Tensor.fromlist(X.shape, X.flatten().tolist())).reshape(100).binary_cross_entropy(Tensor.fromlist(y.shape, y.flatten().tolist()))
    sloss.realize()

    tw0 = torch.tensor(w0.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*w0.shape)
    tb0 = torch.tensor(b0.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*b0.shape)
    tw1 = torch.tensor(w1.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*w1.shape)
    tb1 = torch.tensor(b1.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*b1.shape)

    with torch.no_grad():
      torch_model.inject_weights(tw0,tb0, tw1, tb1)
    torch_model.set_requires_grad()

    torch_model = torch.jit.script(torch_model)
    tout = torch_model(torch.tensor(X, dtype=torch.float32)).reshape(100)
    tloss = torch.nn.functional.binary_cross_entropy(tout, torch.tensor(y, dtype=torch.float32))

    np.testing.assert_allclose(sloss.data(), tloss.detach().numpy(), atol=1e-7, rtol=1e-3)

    sgd = optim.SGD(get_parameters(shrimp_model))
    sgd_ = torch.optim.SGD(torch_model.parameters())
    import time
    @ShrimpJit
    def train_step(X,y): 
      sgd.zero_grad()
      loss = shrimp_model(Tensor.fromlist(X.shape, X.flatten().tolist())).reshape(100).binary_cross_entropy(Tensor.fromlist(y.shape, y.flatten().tolist())).backward()
      sgd.step()
      return loss

    def torch_train_step(X,y):
      sgd_.zero_grad()
      tout = torch_model(torch.tensor(X, dtype=torch.float32)).reshape(100)
      tloss = torch.nn.functional.binary_cross_entropy(tout, torch.tensor(y, dtype=torch.float32))
      tloss.backward()
      sgd_.step()
      return tloss
    for i in range(10):
      s = time.perf_counter()
      sloss = train_step(X,y)
      e_shrimp = time.perf_counter() - s

      s = time.perf_counter()
      tloss = torch_train_step(X,y)
      e_torch = time.perf_counter() - s
      print(f"epoch={i} torch_loss={tloss.detach().numpy()} torch_time={e_torch*1000} shrimp_loss={sloss.data()} shrimp_time={e_shrimp*1000}ms")
      np.testing.assert_allclose(w0.grad.numpy(), torch_model.linear_relu_stack[0].weight.grad.detach().numpy(), atol=1e-1, rtol=1e-1)
      np.testing.assert_allclose(b0.grad.numpy(), torch_model.linear_relu_stack[0].bias.grad.detach().numpy(), atol=1e-1, rtol=1e-1)
      np.testing.assert_allclose(sloss.data(), tloss.detach().numpy(), atol=1e-1, rtol=1e-1)

  def test_basic_net_training_with_adam(self):
    X, y  = dataset()
    y = y.reshape(100)
    # Build the models, realize shrimp model to transfer the weights to the torch model
    shrimp_model = ShrimpModel()
    torch_model = TorchModel()

    w0 = shrimp_model.layers[0].w
    b0 = shrimp_model.layers[0].bias
    w1 = shrimp_model.layers[2].w
    b1 = shrimp_model.layers[2].bias

    sloss = shrimp_model(Tensor.fromlist(X.shape, X.flatten().tolist())).reshape(100).binary_cross_entropy(Tensor.fromlist(y.shape, y.flatten().tolist()))
    sloss.realize()

    tw0 = torch.tensor(w0.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*w0.shape)
    tb0 = torch.tensor(b0.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*b0.shape)
    tw1 = torch.tensor(w1.data().copy(), dtype=torch.float32, requires_grad=True).reshape(*w1.shape)
    tb1 = torch.tensor(b1.data().copy(), dtype=torch.float32,requires_grad=True).reshape(*b1.shape)

    with torch.no_grad():
      torch_model.inject_weights(tw0,tb0, tw1, tb1)
    torch_model.set_requires_grad()

    torch_model = torch.jit.script(torch_model)
    tout = torch_model(torch.tensor(X, dtype=torch.float32)).reshape(100)
    tloss = torch.nn.functional.binary_cross_entropy(tout, torch.tensor(y, dtype=torch.float32))

    np.testing.assert_allclose(sloss.data(), tloss.detach().numpy(), atol=1e-7, rtol=1e-3)

    sgd = optim.Adam(get_parameters(shrimp_model))
    sgd_ = torch.optim.Adam(torch_model.parameters())
    import time
    @ShrimpJit
    def train_step(X,y): 
      sgd.zero_grad()
      loss = shrimp_model(Tensor.fromlist(X.shape, X.flatten().tolist())).reshape(100).binary_cross_entropy(Tensor.fromlist(y.shape, y.flatten().tolist())).backward()
      sgd.step()
      return loss

    def torch_train_step(X,y):
      sgd_.zero_grad()
      tout = torch_model(torch.tensor(X, dtype=torch.float32)).reshape(100)
      tloss = torch.nn.functional.binary_cross_entropy(tout, torch.tensor(y, dtype=torch.float32))
      tloss.backward()
      sgd_.step()
      return tloss

    for i in range(10):
      s = time.perf_counter()
      sloss = train_step(X,y)
      e_shrimp = time.perf_counter() - s

      s = time.perf_counter()
      tloss = torch_train_step(X,y)
      e_torch = time.perf_counter() - s
      print(f"epoch={i} torch_loss={tloss.detach().numpy()} torch_time={e_torch*1000} shrimp_loss={sloss.data()} shrimp_time={e_shrimp*1000}ms")
      np.testing.assert_allclose(w0.grad.numpy(), torch_model.linear_relu_stack[0].weight.grad.detach().numpy(), atol=1e-1, rtol=1e-1)
      np.testing.assert_allclose(b0.grad.numpy(), torch_model.linear_relu_stack[0].bias.grad.detach().numpy(), atol=1e-1, rtol=1e-1)
      np.testing.assert_allclose(sloss.data(), tloss.detach().numpy(), atol=1e-1, rtol=1e-1)

  def test_basic_net_(self):
    weights_shrimp, weights_torch = prepare_tensors([(2,2),(2,2),(2,), (2,)])

    w0, w0_ = weights_shrimp[0], weights_torch[0]
    b0, b0_= weights_shrimp[2], weights_torch[2]
    w1, w1_ = weights_shrimp[1], weights_torch[1]
    b1, b1_ = weights_shrimp[3], weights_torch[3]

    X, _ = dataset()

    x0, x0_ = Tensor.fromlist(X.shape, X.flatten().tolist()), torch.tensor(X, dtype=torch.float32)
    z0 = (x0.dot(w0.transpose()) + b0).relu()
    z0.realize()
    z0_ = (torch.matmul(x0_, w0_.transpose(0,1)) + b0_ ).relu()
    np.testing.assert_allclose(z0.data(), z0_.detach().numpy(), atol=1e-6, rtol=1e-3)

    z1 = (z0.dot(w1.transpose()) + b1).sigmoid().square().mean()
    z1.realize()
    z1_ = (torch.matmul(z0_, w1_.transpose(0,1)) + b1_ ).sigmoid().square().mean()
    np.testing.assert_allclose(z1.data(), z1_.detach().numpy(), atol=1e-6, rtol=1e-2)

    z1.backward()
    z1_.backward()
    np.testing.assert_allclose(w0.grad.numpy(), w0_.grad.detach().numpy(), atol=1e-4, rtol=1e-2)

  def test_basic_net_bce_loss_(self):
    weights_shrimp, weights_torch = prepare_tensors([(2,2),(1,2),(2,), (1,)])

    w0, w0_ = weights_shrimp[0], weights_torch[0]
    b0, b0_= weights_shrimp[2], weights_torch[2]
    w1, w1_ = weights_shrimp[1], weights_torch[1]
    b1, b1_ = weights_shrimp[3], weights_torch[3]

    X, y = dataset()

    x0, x0_ = Tensor.fromlist(X.shape, X.flatten().tolist()), torch.tensor(X, dtype=torch.float32)
    z0 = (x0.dot(w0.transpose()) + b0).relu()
    z0.realize()
    z0_ = (torch.matmul(x0_, w0_.transpose(0,1)) + b0_ ).relu()
    np.testing.assert_allclose(z0.data(), z0_.detach().numpy(), atol=1e-6, rtol=1e-3)

    z1 = (z0.dot(w1.transpose()) + b1).sigmoid()
    z1.realize()
    z1_ = (torch.matmul(z0_, w1_.transpose(0,1)) + b1_ ).sigmoid()
    np.testing.assert_allclose(z1.data(), z1_.detach().numpy(), atol=1e-6, rtol=1e-3)

    z2 = z1.reshape(100).binary_cross_entropy(_:=Tensor.fromlist(y.shape, y.flatten().tolist()))
    z2.realize()
    z2_ = torch.nn.functional.binary_cross_entropy(z1_.reshape(100), torch.tensor(y, dtype=torch.float32))
    np.testing.assert_allclose(z2.data(), z2_.detach().numpy(), atol=1e-6, rtol=1e-3)

    z2.backward()
    z2_.backward()

    np.testing.assert_allclose(w0.grad.numpy(), w0_.grad.detach().numpy(), atol=1e-4, rtol=1e-3)
  
  def test_batchnorm2d(self, training=True, threed=False):
    # From shrimp
    if training: 
      szs = [4, 8, 16, 32]
      for sz in szs:
        # create in shrimp 
        bn = BatchNorm(sz, eps=1e-5, track_running_stats=training)
        bn.weight = Tensor.randn(sz)
        bn.bias = Tensor.randn(sz)
        bn.running_mean = Tensor.randn(sz)
        bn.running_var = Tensor.randn(sz)
        bn.running_var.numpy()[bn.running_var.numpy() < 0] = 0

        # create in torch
        with torch.no_grad():
          if threed:
            tbn = torch.nn.BatchNorm3d(sz).eval()
          else:
            tbn = torch.nn.BatchNorm2d(sz).eval()
          tbn.training = training
          tbn.weight[:] = torch.tensor(bn.weight.numpy())
          tbn.bias[:] = torch.tensor(bn.bias.numpy())
          tbn.running_mean[:] = torch.tensor(bn.running_mean.numpy())
          tbn.running_var[:] = torch.tensor(bn.running_var.numpy())

        np.testing.assert_allclose(bn.running_mean.numpy(), tbn.running_mean.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.detach().numpy(), rtol=1e-5, atol=1e-6)

        # trial
        if threed:
          inn = Tensor.randn(2, sz, 3, 3, 3)
        else:
          inn = Tensor.randn(2, sz, 3, 3)

        outt = bn(inn, training=True)

        # in torch
        toutt = tbn(torch.tensor(inn.numpy()))

        # close
        np.testing.assert_allclose(outt.numpy(), toutt.detach().numpy(), rtol=5e-4, atol=1e-6)
        np.testing.assert_allclose(bn.running_mean.numpy(), tbn.running_mean.detach().numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bn.running_var.numpy(), tbn.running_var.detach().numpy(), rtol=1e-5, atol=1e-6)

  def test_layer_norm(self):
    # Create in shrimp
    ln = LayerNorm(sz:=(2,))
    ln.weight = Tensor.randn(*sz)
    ln.bias = Tensor.randn(*sz)

    with torch.no_grad():
      torch_ln = torch.nn.LayerNorm([2])
      torch_ln.weight[:] = torch.tensor(ln.weight.numpy())
      torch_ln.bias[:] = torch.tensor(ln.bias.numpy())

    x = Tensor.randn(2,2)
    xt = torch.tensor(x.numpy())

    sln = ln(x)
    tln = torch_ln(xt)

    np.testing.assert_allclose(sln.numpy(), tln.detach().numpy(), rtol=1e-3, atol=1e-3)

  def test_layer_norm_with_more_dims(self):
    ln = LayerNorm(sz:=(2,3,4))
    ln.weight = Tensor.randn(*sz)
    ln.bias = Tensor.randn(*sz)

    with torch.no_grad():
      torch_ln = torch.nn.LayerNorm([2,3,4])
      torch_ln.weight[:] = torch.tensor(ln.weight.numpy())
      torch_ln.bias[:] = torch.tensor(ln.bias.numpy())

    x = Tensor.randn(4,5,2,3,4)
    xt = torch.tensor(x.numpy())

    sln = ln(x)
    tln = torch_ln(xt)

    np.testing.assert_allclose(sln.numpy(), tln.detach().numpy(), rtol=1e-3, atol=1e-3)
  
  def test_layer_norm_shape_wrong_assert_raise(self):
    with self.assertRaises(AssertionError):
      ln = LayerNorm(sz:=(2,3,4))
      ln.weight = Tensor.randn(*sz)
      ln.bias = Tensor.randn(*sz)
      ln(Tensor.randn(5,3,2,3,5))
  
  def test_layer_norm_shape_correct_no_assert_raise(self):
    ln = LayerNorm(sz:=(2,3,4))
    ln.weight = Tensor.randn(*sz)
    ln.bias = Tensor.randn(*sz)
    ln(Tensor.randn(5,3,2,3,4))
  
  def test_layer_norm_norm_shape_singleton_norms_last_dim_size_matches(self):
    ln = LayerNorm(1, elementwise_affine=False)
    out = ln(Tensor.randn(2,2,1)).numpy()
    assert out.shape == (2,2,1)
  
  def test_layer_norm_norm_shape_singleton_last_dim_doesnt_match_raises(self):
    with self.assertRaises(AssertionError):
      ln = LayerNorm(2, elementwise_affine=False)
      out = ln(Tensor.randn(2,2,1)).numpy()
      assert out.shape == (2,2,1)
  
  def test_conv2d(self):
    # Create in shrimp
    conv = nn.Conv2D(in_channels=3, out_channels=6, kernel_size=(3,3))
    x = Tensor.randn(1,3,5,5)
    out = conv(x)
    assert out.shape == (1,6,3,3) 

    # Create in torch
    with torch.no_grad():
      torch_conv = torch.nn.Conv2d(3,6,(3,3))
      assert torch_conv.bias is not None
      assert conv.bias is not None
      torch_conv.weight[:] = torch.tensor(conv.weight.numpy())
      torch_conv.bias[:] = torch.tensor(conv.bias.numpy())

    xt = torch.tensor(x.numpy())
    outt = torch_conv(xt)
    np.testing.assert_allclose(out.numpy(), outt.detach().numpy(), rtol=1e-3, atol=1e-3)

    # Test backward pass
    out.sum().backward().realize()
    outt.sum().backward()

    # Check gradients for weights and bias
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None
    assert torch_conv.weight.grad is not None
    assert torch_conv.bias.grad is not None
    np.testing.assert_allclose(conv.weight.grad.numpy(), torch_conv.weight.grad.numpy(), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(conv.bias.grad.numpy(), torch_conv.bias.grad.numpy(), rtol=1e-3, atol=1e-3)

  def test_mnist_loader(self):
    train_images, train_labels, test_images, test_labels = mnist_loader(10)
    assert train_images.shape == (10, 1, 28, 28)
    assert train_labels.shape == (10,)
    assert test_images.shape == (10, 1, 28, 28)
    assert test_labels.shape == (10,)

  def test_conv2d_mnist(self):
    train_images, train_labels, test_images, test_labels = mnist_loader(10)
    model = ConvNet()
    out = model(train_images)
    # TODO: We need to add proper dtype promotion for binary operations such that we use
    # the dtype that can actually store the value of the computation rather than defaulting to
    # the dtype of the input A 

