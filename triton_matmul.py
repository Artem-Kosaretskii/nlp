import torch
import triton
import triton.language as tl

@triton.jit
def tiled_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)


    a_tile_ptr = a_ptr + pid_m * BLOCK_SIZE_M * stride_am
    b_tile_ptr = b_ptr + pid_n * BLOCK_SIZE_N * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a_block = tl.load(a_tile_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_am +
                          tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak,
                          mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < BLOCK_SIZE_M) &
                               (tl.arange(0, BLOCK_SIZE_K)[None, :] < BLOCK_SIZE_K))

        b_block = tl.load(b_tile_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk +
                          tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_bn,
                          mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < BLOCK_SIZE_K) &
                               (tl.arange(0, BLOCK_SIZE_N)[None, :] < BLOCK_SIZE_N))

        accumulator += tl.dot(a_block, b_block)

        a_tile_ptr += BLOCK_SIZE_K * stride_ak
        b_tile_ptr += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + pid_m * BLOCK_SIZE_M * stride_cm + pid_n * BLOCK_SIZE_N * stride_cn
    tl.store(c_ptrs, accumulator)
