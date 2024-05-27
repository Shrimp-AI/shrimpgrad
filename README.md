# ShrimpGrad - Yet Another Tensor Library

## Warning:
You can't do much right now except build the compute graph and get a list of kernels that need to be code generated and compiled. We did have end-to-end execution, you can see this in examples, but it was just python looping over lists for testing purposes.

All the op tests now fail since we removed the python runtime in order to migrate to autogenerating kernels. Since no one is using it, it doesn't matter.

### Goals
1. Lower the scheduled kernels to the linearized IR
2. Generate C code from the linearized IR
3. Execute this code
4. Add CUDA codegen (repeat)

## What?

A python library to create tensor operations in python and automatically compile and execute them on an accelerator. Use the backward pass to compute gradients, and an optimizer to train your model.

```python
from shrimpgrad import Tensor

x = Tensor.ones((2,2))
y = Tensor.zeros((2,2))
z = x.dot(y)
z.backward()
```

## Why YATL?

With so many tensor libraries already in mainstream use, what is the point of yet another tensor library? To learn. And to teach. And maybe build something better than what exists.

## Why is it like tinygrad?

Because we are learning the concepts of tinygrad to figure out if they can be expressed in a better way towards the same goal.

## Install

```
git clone https://github.com/Shrimp-AI/shrimpgrad.git
cd shrimpgrad
python3 -m pip install .
```
