# File: benchmarks/benchmark_attention.py
import torch
import triton
from rl_engine.kernels.ops.attention import prefix_shared_sdpa

def run_benchmark():
    bs = 1
    G = 64
    len_q = 512
    len_kv = 4096
    dim = 128

    print(f"Benchmarking GRPO Prefix-Shared Attention")
    print(f"Shape: bs={bs}, G={G}, len_q={len_q}, len_kv={len_kv}, dim={dim}")

    q = torch.randn(bs, G, len_q, dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")

    out_ref = prefix_shared_sdpa(q, k, v)
    out_custom = prefix_shared_sdpa(q, k, v)
    # torch.testing.assert_close(out_custom, out_ref, atol=1e-2, rtol=1e-2)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['len_kv'],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg='provider',
            line_vals=['torch_native', 'rl_kernel_shared'],
            line_names=['PyTorch Native SDPA', 'RL-Kernel Prefix-Shared'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='Latency (ms)',
            plot_name='prefix-shared-attention-performance',
            args={'bs': bs, 'G': G, 'len_q': len_q, 'dim': dim}
        )
    )
    def benchmark(bs, G, len_q, len_kv, dim, provider):
        q = torch.randn(bs, G, len_q, dim, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(bs, len_kv, dim, dtype=torch.bfloat16, device="cuda")

        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch_native':
            k_exp = k.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim).contiguous()
            v_exp = v.unsqueeze(1).expand(-1, G, -1, -1).reshape(bs * G, 1, len_kv, dim).contiguous()
            q_res = q.view(bs * G, 1, len_q, dim)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(q_res, k_exp, v_exp), 
                quantiles=quantiles
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: prefix_shared_sdpa(q, k, v), 
                quantiles=quantiles
            )
        return ms, min_ms, max_ms

    benchmark.run(print_data=True)

if __name__ == "__main__":
    run_benchmark()