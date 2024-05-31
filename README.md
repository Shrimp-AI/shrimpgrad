# ShrimpGrad - Yet Another Tensor Library

# Architecture
tensor.py (Key object Tensor)
- A front-end that behaves like pytorch allowing easy model creation and optimization.
```python
from shrimpgrad import Tensor

x = Tensor.ones((2,2))
y = Tensor.zeros((2,2))
z = x.dot(y)
z.backward()
```
function.py
- Generates the Thunk AST of all the operations on tensors preserving important information but not executing them
  - forward and backward passes are codified here

future.py
- Defines the Thunk (a Haskell concept of computations that are not evaluated until they are needed).
- Thunks store their operands (other thunks), their base (if they are a result of a movement op), and their views
- Maintain a buffer if they are base thunks (buffers are used by the engine to actually store the data for consumption by accelerators
- IndexedForwardGraph - extract as much in from the Thunk AST and store it for optimization passes

schedule.py
- Figure out what needs to be scheduled first. Most likely LoadOp Thunk to make sure the data is in the buffer before we execute any code on the acceleartor.
- Call into the fusion engine to reduce the graph
- Generate a mid-level intermediate representation for the lowering engine (MLIR -> LLIR) and define which inputs and outputs correspond to a MLIR AST

fuse_ops.py
- Given an IndexedForwardGraph compute a immediate post dominator tree and fuse operations that can be fused according to fuse conditions
- Injective operators can fuse with child Injective operators
- There can only be one Reduction operator per fused kernel and it must be the last operator
  - i.e.) you can fuse many injectives in a row followed by a single reduction
- Checks if a node can fuse with its immediate post dominator (the node for which all outputs of this node must pass through)
  - if it can fuse with the ipdom then it traverses the path between the src and the immediate post dominator sink and checks
    that fuse conditions are not violated
  - if successfull all nodes including source and sink are grouped via union find

postdomtree.py
- Given an IndexedForwardGraph compute all immediate post dominators and group accordingly
- Since ASTs are DAGs we compute the LCA of all output nodes to determine the ipdom


## Goals
1. Generate Fused Kernels using LCA and IndexedForwardGraphs to compute immediate post dominators
2. Lower the scheduled kernels to the linearized IR
3. Generate C code from the linearized IR
4. Execute this code
5. Add other accelerators
6. Optimization 

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

With so many tensor libraries already in mainstream use, what is the point of yet another tensor library? Build something better than what exists. Make custom accelerator chips.

## Install

```
git clone https://github.com/Shrimp-AI/shrimpgrad.git
cd shrimpgrad
python3 -m pip install .
```
