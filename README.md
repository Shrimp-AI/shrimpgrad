# ShrimpGrad - Yet Another Tensor Library
## What  is ShrimpGrad?

A simple, minimalist, lazily evaluated, JIT-able tensor library for modern deep learning.

```python
from shrimpgrad import Tensor, nn
from shrimpgrad.engine.jit import ShrimpJit
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.1)
X = X.astype(float)
y = y.astype(float)

class ShallowNet:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Linear(2, 50), Tensor.relu,
      nn.Linear(50, 1), Tensor.sigmoid,
    ]
  def __call__(self, x: Tensor):
    return x.sequential(self.layers)

@ShrimpJit
def train(X,y):
  sgd.zero_grad()
  out = model().reshape(100)
  loss = out.binary_cross_entropy()
  loss.backward()
  sgd.step()
  return out, loss

X = Tensor.fromlist(X.shape, X.flatten().tolist())
y = Tensor.fromlist(y.shape, y.flatten().tolist())
for epoch in range(50): train(X,y)
```
## RISC Inspired
A reduced set of "instructions" is needed to define everything from matrix multiplication to 2D convolutions

1. Binary - ADD, MUL, DIV, MAX, MOD, CMPLT, COMPEQ, XOR
2. Unary - EXP2, LOG2, CAST, SIN, SQRT, NEG
3. Ternary - WHERE
4. Reduce - SUM, MAX
5. Movement - RESHAPE, PERMUTE, EXPAND, PAD, SHRINK
6. Load - EMPTY, COPY, CONST, ASSIGN

## Easy JIT Compilation
Go full native with ease. JIT lowers execution from python to the accelerator including forward/backward passes and optimizer steps.

## Install
The easiest way to get going is to install [nix](https://nixos.org/download/).
```
git clone https://github.com/Shrimp-AI/shrimpgrad.git
cd shrimpgrad
nix-shell
```
Otherwise

```
python3 -m pip install -e '.[testing]'
```
