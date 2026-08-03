import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    num_stages = 3
    grid = (n_rows,)
    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages
    )
    return y


def test_softmax_correctness():
    torch.manual_seed(42)
    x = torch.randn(100, 256, device='cuda')
    y_torch = torch.softmax(x, dim=-1)
    y_triton = triton_softmax(x)
    max_diff = torch.max(torch.abs(y_torch - y_triton)).item()
    print(f"Max diff: {max_diff:.8f}")
    assert max_diff < 1e-6, f"Very big difference: {max_diff}"
    print("Results match")
    print("\nExtremes...")

    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]], device='cuda')
    y_torch_large = torch.softmax(x_large, dim=-1)
    y_triton_large = triton_softmax(x_large)

    diff_large = torch.max(torch.abs(y_torch_large - y_triton_large)).item()
    print(f"Big numbers - difference: {diff_large:.8f}")
    assert diff_large < 1e-6

    x_small = torch.tensor([[-1000.0, -1001.0, -1002.0]], device='cuda')
    y_torch_small = torch.softmax(x_small, dim=-1)
    y_triton_small = triton_softmax(x_small)

    diff_small = torch.max(torch.abs(y_torch_small - y_triton_small)).item()
    print(f"Small numbers - difference: {diff_small:.8f}")
    assert diff_small < 1e-6


def benchmark_softmax():
    sizes = [
        (128, 64),
        (1024, 256),
        (4096, 1024),
        (10000, 2048),
    ]

    for n_rows, n_cols in sizes:
        print(f"\nSize: {n_rows}x{n_cols}")
        x = torch.randn(n_rows, n_cols, device='cuda')
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            y_torch = torch.softmax(x, dim=-1)
        end.record()
        torch.cuda.synchronize()
        torch_time = start.elapsed_time(end) / 100

        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            y_triton = triton_softmax(x)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 100

        max_diff = torch.max(torch.abs(y_torch - y_triton)).item()

        print(f"PyTorch: {torch_time:.4f} ms")
        print(f"Triton:  {triton_time:.4f} ms")
        print(f"Acceleration: {torch_time / triton_time:.2f}x")
        print(f"Error: {max_diff:.8f}")


if __name__ == "__main__":
    test_softmax_correctness()
    benchmark_softmax()

    print("\nAll done")
