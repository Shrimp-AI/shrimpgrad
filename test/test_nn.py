from shrimpgrad import Tensor, nn 
import unittest
import torch
import numpy as np

class TestNN(unittest.TestCase):
  def test_linear(self):
    x = Tensor((2,2), [1.0,2.0,3.0,4.0])
    model = nn.Linear(2,2)
    z = model(x)
    with torch.no_grad():
      torch_model = torch.nn.Linear(2,2).eval()
      torch_model.weight[:] = torch.tensor(model.w.data).reshape(2,2)
      torch_model.bias[:] = torch.tensor(model.bias.data)
      torch_x = torch.tensor(x.data).reshape(2,2)
      torch_z = torch_model(torch_x)
    np.testing.assert_allclose(np.array(z.data).reshape(2,2), torch_z.detach().numpy(), atol=5e-4, rtol=1e-5)
