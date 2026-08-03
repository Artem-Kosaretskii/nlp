import torch
import triton
import triton.language as tl


@triton.jit
def dot_product_simple_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    block_sum = tl.sum(x * y, axis=0)
    tl.atomic_add(output_ptr, block_sum)


def dot_product_full_gpu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.dim() == 1

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    result = torch.zeros(1, dtype=torch.float32, device='cuda')
    dot_product_simple_kernel[(num_blocks,)](x, y, result, n_elements, BLOCK_SIZE)

    return result.cpu()


if __name__ == "__main__":
    size = 10000
    x = torch.arange(1, size + 1, dtype=torch.float32, device='cuda')
    y = torch.ones(size, dtype=torch.float32, device='cuda') * 2

    result_simple = dot_product_full_gpu(x, y)
    result_pytorch = torch.dot(x, y).cpu()

    print(f"Triton: {result_simple.item():.1f}")
    print(f"PyTorch: {result_pytorch.item():.1f}")
    print(f"Matching: {torch.allclose(result_simple, result_pytorch, atol=1e-4)}")

    import time

    print("\nBenchmark:")
    start = time.time()
    for _ in range(100):
        result = dot_product_full_gpu(x, y)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    start = time.time()
    for _ in range(100):
        result = torch.dot(x, y)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start

    print(f"Triton: {triton_time:.4f}с")
    print(f"PyTorch: {pytorch_time:.4f}с")
    print(f"Ratio: {pytorch_time / triton_time:.2f}x")
