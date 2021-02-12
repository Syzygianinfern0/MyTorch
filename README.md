<div align="center">

# 🔥 MyTorch 🔥
🐣 A Library extending PyTorch for Personal Needs backed by C++/CUDA APIs 

| **🚧 WIP Forever 🚧** |
|:-------------------:|

---

</div>

# Documentation 📑

## [`mytorch.ops`](https://github.com/Syzygianinfern0/MyTorch/tree/main/mytorch)

### [`mytorch.ops.im2col`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/ops/im2col.py) and [`mytorch.ops.col2im`](https://github.com/Syzygianinfern0/MyTorch/blob/main/mytorch/ops/im2col.py)
- Rearrange image blocks into columns.
- The representation is used to perform GEMM-based convolution.
- Output is 5D (or 6D in case of minibatch) tensor.
- Minibatch implementation is inefficient, and could be done in a single CUDA kernel.
