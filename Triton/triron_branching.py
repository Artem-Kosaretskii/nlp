import torch
import triton
import triton.language as tl
import time

@triton.jit
def good_branching_kernel(
        input_ptr, output_ptr,
        N,
        threshold,
        BLOCK_SIZE,
        stride_in, stride_out,
):
    pid = tl.program_id(0)
    indices = pid * 256 + tl.arange(0, 256)
    data = tl.load(input_ptr + indices * stride_in)
    mask_above = data > threshold
    sqrt_result = tl.sqrt(data) + 1
    square_result = data * data - 2
    result = tl.where(mask_above, sqrt_result, square_result)
    tl.store(output_ptr + indices * stride_out, result)


def good_branching_operation(x: torch.Tensor, thr: float):
    assert x.is_cuda, "Tensor must be on GPU"
    assert x.dim() == 1, "Тensor must be a vector"

    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    result = torch.zeros(x.shape[0], dtype=torch.float32, device='cuda')
    stride_in = x.stride(0)
    stride_out = result.stride(0)
    good_branching_kernel[(num_blocks,)](x, result, n_elements, thr, BLOCK_SIZE, stride_in, stride_out)

    return result


def torch_branching_operation(x: torch.Tensor, thr: float):
    assert x.is_cuda, "Tensor must be on GPU"
    assert x.dim() == 1, "Тensor must be a vector"
    mask_above = x > thr
    sqrt_result = torch.sqrt(x) + 1
    square_result = x * x - 2
    result = torch.where(mask_above, sqrt_result, square_result)

    return result


def main():

    x = torch.tensor([-2.0, -1.0, 0.0, 0.3, 0.7, 1.0, 2.0], device='cuda')
    threshold = 0.5

    print("Input:", x.cpu().numpy())
    print("Threshold:", threshold)

    result_good = good_branching_operation(x, threshold)
    result_torch = torch_branching_operation(x, threshold)

    print("Triton:", result_good.cpu().numpy())
    print("PyTorch:", result_torch.cpu().numpy())

    try:
        torch.testing.assert_close(result_good, result_torch, rtol=1e-6, atol=1e-6)
        print("Correct")
    except AssertionError as e:
        print("Incorrect:", e)

    x_large = torch.randn(1000000, device='cuda')

    torch.cuda.synchronize()
    start = time.time()
    _ = torch_branching_operation(x_large, threshold)
    torch.cuda.synchronize()
    torch_time = time.time() - start

    start = time.time()
    _ = good_branching_operation(x_large, threshold)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    print(f"\nProductivity test:")
    print(f"Triton: {triton_time:.7f}s")
    print(f"Torch: {torch_time:.7f}s")
    print(f"Ratio: {torch_time / triton_time:.2f}x")


if __name__ == '__main__':
    main()
