<div align="center">

# 🔥 MyTorch 🔥
🐣 A Library extending PyTorch for Personal Needs backed by C++/CUDA APIs 

| **🚧 WIP Forever 🚧** |
|:-------------------:|

---

</div>

# Installation 👨‍💻
I have not included any dependencies in the `setup.py` nor a `requirements.txt` as I leave the hassle of setting up GPU support for torch on your own. It should work on `torch>=1.4` and `CUDA>=10.0` but I frankly have no clue. I use `torch==1.7.1` and `CUDA` Version of `11.2`

To install it, just do
```shell
pip install git+https://github.com/Syzygianinfern0/MyTorch.git
```

Its also available on PyPi, but I wouldn't be very keen on maintaining it. 

```shell
pip install python-mytorch==0.1
```

# Documentation 📑

## [`mytorch.ops`](https://github.com/Syzygianinfern0/MyTorch/tree/main/mytorch/ops)

### [`mytorch.ops.im2col`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/ops/im2col.py) and [`mytorch.ops.col2im`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/ops/im2col.py)
- Rearrange image blocks into columns.
- The representation is used to perform GEMM-based convolution.
- Output is 5D (or 6D in case of minibatch) tensor.
- Minibatch implementation is inefficient, and could be done in a single CUDA kernel.

## [`mytorch.layers`](https://github.com/Syzygianinfern0/MyTorch/tree/main/mytorch/layers)

### [`mytorch.layers.DeformConv`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/layers/deform_conv_module.py) and [`mytorch.layers.DeformUnfold`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/layers/deform_unfold_module.py)
- JIT Implementation of Deformable Convolution and Deformable Unfolding

### [`mytorch.layers.ReDynamicWeightsCat33`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/layers/dynamic_weights.py)
- JIT Implementation of 3x3 Dynamic Graph Message Passing Convolution