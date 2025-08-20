import torch
import triton
import triton.language as tl

@triton.jit
def softmax(outputPtr, inputPtr, 
            stride_output, stride_input, 
            n_cols, BLOCK_SIZE: tl.constexpr):
  row_idx = tl.program_id(0)
  row_start_ptr = inputPtr + row_idx * stride_input
  out_row_start_ptr = outputPtr + row_idx * stride_output

  offsets = tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_cols

  row = tl.load(row_start_ptr + offsets, mask = mask, other = -float("inf"))

  row_max = tl.max(row, axis = 0)
  row = row - row_max
  numerator = tl.exp(row)
  denominator = tl.sum(numerator, axis = 0)
  softmax_row = numerator / denominator
  tl.store(out_row_start_ptr + offsets, softmax_row, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
  assert x.is_cuda, "Input must be on CUDA"
  B,N = x.shape
  y = torch.empty_like(x)

  BLOCK_SIZE = triton.next_power_of_2(N)
  softmax[(B,)](
      y,x,
      y.stride(0), x.stride(0),
      N,
      BLOCK_SIZE=BLOCK_SIZE
  )
  return y
