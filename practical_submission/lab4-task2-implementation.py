import torch
import torch.nn.functional as F
import time

def timed_gpu(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def timed_cpu(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start

def get_sdpa_data(batch=32, heads=8, seq_len=128, embed_dim=64, device="cpu"):
    Q = torch.randn(batch, heads, seq_len, embed_dim, device=device)
    K = torch.randn(batch, heads, seq_len, embed_dim, device=device)
    V = torch.randn(batch, heads, seq_len, embed_dim, device=device)
    return Q, K, V

# Naive SDPA implementation
def naive_sdpa(Q, K, V):
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # GEMM 1
    attn_probs = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output

# Optimized SDPA using PyTorch fused kernel
def fused_sdpa(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

# Function to measure SDPA execution time
def time_sdpa(fn, n=100, device="cpu"):
    """
    Measures the average execution time of an SDPA function.
    """
    Q, K, V = get_sdpa_data(device=device)
    
    # Warm-up to stabilize results
    for _ in range(5):
        fn(Q, K, V)

    # Measure execution time
    times = []
    for _ in range(n):
        if device == "cpu":
            _, t = timed_cpu(lambda: fn(Q, K, V))
        else:
            _, t = timed_gpu(lambda: fn(Q, K, V))
        times.append(t)

    avg_time = sum(times) / len(times)
    return avg_time

# Device
device = "cpu" # or cuda

print(f"Running SDPA profiling on {device.upper()}...\n")

n_iters = 200

# Baseline Naive SDPA
naive_time = time_sdpa(naive_sdpa, n=n_iters, device=device)
print(f"Naive SDPA avg time: {naive_time:.6f} sec")

# Optimized SDPA
fused_time = time_sdpa(fused_sdpa, n=n_iters, device=device)
print(f"Fused SDPA avg time: {fused_time:.6f} sec")

# Compare speedup
if fused_time < naive_time:
    print(f"Fused SDPA is {naive_time / fused_time:.2f}x faster than Naive SDPA!")
else:
    print(f"No speedup observed. Possible reasons:")
    print("- Compilation overhead for small batch sizes.")
    print("- Already optimized memory access on certain hardware.")
