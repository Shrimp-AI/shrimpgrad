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

With so many tensor libraries already in mainstream use, what is the point of yet another tensor library? We are not sure but something stinks.

And it's the polluted mainstream. 

PyTorch is open source but belongs to a behemoth corporation. They don't care about NVIDIA's domination in the accelerator market, and if they do care it's only because they want to dominate it. They'll just build their own accelerators and garble pytorch etc. into more of an overengineered mess using its legions of "open source developers". 

On the fringes of tensor libs sits tinygrad. The premise of tinygrad is to autogenerate kernel code and execute it for you. There are no high level primitives like matmul with custom coded kernels in CUDA. Everything is constructed from a reduced instruction set architecture (lol). Tensor operations are all defined from these "ops". So the compute graph is general and therefore can be lowered to an intermediate representation that is used to generate kernel code for various accelerators.


## Why is it like tinygrad?

We are improving on the design of tinygrad. Right now things look "similar". Everything is coded from the perspective of reverse engineering tinygrad. Basically to understand tinygrad end to end. Once we understand the design we can make it better, and since we are not focused on satisfying investors, there's no agenda to say "make it run on AMD" as a proof of concept before we "tape out chips". 

Tinygrad's codebase is "tiny" for tinys sake and therefore hard to understand; with limited documentation for the heady parts like lowering from LazyBuffers to UOps via scheduling and linearization. The management of shapes via movement operations also has a lot of code that is hard to follow because the reasons for a lot of the choices are unclear (like the massive amount of lines dedicated to masking). One wonders if the ends justify the means. It's attempt to be smart for smartness bothers our feeble minds. Maybe because we are idiots. 

In short, we are spending our precious and limited time on Earth stepping through tinygrad to expose the underlying algorithms. Then we'll see what remains standing in the end. It could be tinygrad expresses the ideas in the optimal manner and there's nothing to do except make it cleaner. A clean rewrite at worst, a better library at best.

## Install

```
git clone https://github.com/Shrimp-AI/shrimpgrad.git
cd shrimpgrad
python3 -m pip install .
```
