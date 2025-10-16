# I'm pretty sure this works on ROCm now.

This will probably only work correctly with the latest ROCm/Torch nightlies as of the time of writing. This didn't work the last time I tried building this, but now it does, probably due to a [recent Torch Hipify change](https://github.com/pytorch/pytorch/commit/ae61b7f560175633d91df57d8b144b2f71ed41b6). I'm using TheRock nightlies on Linux with a GFX11/RDNA3 GPU so YMMV, but this builds for your native architecture and should work fine for any AMD GPU with hipBLASLt support. Follow the Installation section below to install.

# CublasOps: High-Performance Linear Layers with cuBLAS and cuBLASLt

CublasOps is a PyTorch extension library that provides high-performance linear layers for half-precision (FP16) matrix multiplications using NVIDIA's cuBLAS and cuBLASLt libraries. It offers fast and efficient execution of `A x B^T` matrix multiplications with optional bias addition and activation functions (ReLU or GELU).

## Features

- Fast half-precision (FP16) matrix multiplications using cuBLAS and cuBLASLt with half precision accumulation (2x speedup on 4090).
- Support for fused operations: matmul + bias + activation (ReLU or GELU)
- Easy-to-use linear layers: `CublasLinear`, `CublasLinearRelu`, and `CublasLinearGelu`
- Seamless integration with PyTorch models
- Batched and non-batched operations

For example:
using the cublas linear with 4096, 4096 in/out features, and a (2, 4096, 4096) input tensor on my RTX 4090:
```
CUBLAS INFERENCE: 
FLOPS: 274877906944
TFLOP/s: 305.801

TORCH INFERENCE: 
FLOPS: 274877906944
TFLOP/s: 166.989
```


## Installation

To install CublasOps, follow these steps:

1. Make sure you have PyTorch installed with CUDA support.
2. Clone the repository:
   ```
   git clone https://github.com/gel-crabs/torch-cublas-hgemm-rocm.git
   ```
3. Navigate to the cloned repository:
   ```
   cd torch-cublas-hgemm-rocm
   ```
4. Build and install the extension:
   ```
   python -m pip install -U -v .
   ```

## Usage

Here's a simple example of how to use CublasOps in your PyTorch code:

```python
import torch
from cublas_ops import CublasLinear, CublasLinearGelu, CublasLinearRelu

in_features = 64
out_features = 64
bias = True  # or False

# (A x B^T + bias)
linear = CublasLinear(in_features, out_features, bias=bias, device='cuda', dtype=torch.float16)
input_tensor = torch.randn((2, 8, 64)).cuda().half()
# or...
input_tensor = torch.randn((8, 64)).cuda().half()

output_tensor = linear(input_tensor)

# For fused GELU: gelu(A x B^T + bias)
linear_gelu = CublasLinearGelu(in_features, out_features, bias=bias, device='cuda', dtype=torch.float16)

# For fused ReLU: relu(A x B^T + bias)
linear_relu = CublasLinearRelu(in_features, out_features, bias=bias, device='cuda', dtype=torch.float16)
```

## API Reference

### Linear Layers

- `CublasLinear(in_features, out_features, bias=True, device=None, dtype=torch.float16, epilogue_str="NONE")`
  - A linear layer that performs `A x B^T + bias` matrix multiplication with optional bias addition.
- `CublasLinearGelu(in_features, out_features, bias=True, device=None, dtype=torch.float16)`
  - A linear layer with fused GELU activation: `gelu(A x B^T + bias)`.
- `CublasLinearRelu(in_features, out_features, bias=True, device=None, dtype=torch.float16)`
  - A linear layer with fused ReLU activation: `relu(A x B^T + bias)`.

### Low-Level Functions

- `cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor)`
  - Performs a simple `A x B^T` matrix multiplication using cuBLAS.
- `cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor)`
  - Performs a batched `A x B^T` batched matrix multiplication using cuBLAS. At least one of A/B should have 3 dimensions, with the other having 2 or 3.
- `cublaslt_fused_half_matmul_simple(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None, epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE")`
  - Performs a fused `optional_activation(A x B^T + optional(bias))` matrix multiplication with optional bias addition and activation using cuBLASLt.
- `def cublaslt_fused_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor] = None, epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE")`
  - Performs a fused `optional_activation(A x B^T + optional(bias))` batched matrix multiplication with optional bias addition and activation using cuBLASLt. At least one of A/B should have 3 dimensions, with the other having 2 or 3.

## Contributing

Contributions to CublasOps are welcome! If you encounter any issues, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request on the GitHub repository.

## Acknowledgments

CublasOps is built upon the powerful cuBLAS and cuBLASLt libraries provided by NVIDIA. We would like to thank the NVIDIA team for their excellent work on these libraries.
