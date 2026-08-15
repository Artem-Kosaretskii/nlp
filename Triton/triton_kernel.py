import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
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
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Тензоры должны быть на GPU"
    assert x.shape == y.shape, "Тензоры должны иметь одинаковую форму"
    output = torch.empty_like(x)
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)
    return output


if __name__ == "__main__":
    size = 10000
    x_cpu = torch.arange(1, size + 1, dtype=torch.float32)  # [1, 2, 3, ..., 10000]
    y_cpu = torch.ones(size, dtype=torch.float32) * 5  # [5, 5, 5, ..., 5]
    x_gpu = x_cpu.cuda()
    y_gpu = y_cpu.cuda()

    print("X (first 10):", x_cpu[:10].tolist())
    print("Y (first 10):", y_cpu[:10].tolist())
    result_triton = add_vectors(x_gpu, y_gpu)
    result_pytorch = x_gpu + y_gpu
    print("\nComparing triton and pytorch...")
    is_correct = torch.allclose(result_triton, result_pytorch, atol=1e-6)
    print(f"Matching: {is_correct}")

    print(f"\Result (first 10): {result_triton.cpu()[:10].tolist()}")
    print(f"Expected: {[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}")

    print("\nRunning benchmark...")
    import time
    for _ in range(10):
        _ = add_vectors(x_gpu, y_gpu)
        _ = x_gpu + y_gpu

    start_time = time.time()
    for _ in range(100):
        result_triton = add_vectors(x_gpu, y_gpu)
    torch.cuda.synchronize()
    triton_time = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        result_pytorch = x_gpu + y_gpu
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time

    print(f"Triton: {triton_time:.4f} сек")
    print(f"PyTorch: {pytorch_time:.4f} сек")
    print(f"Ratio: {pytorch_time / triton_time:.2f}x")
