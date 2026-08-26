"""
Verify the Blackwell (sm_120) setup end to end.

Checks device identity, confirms the PyTorch build actually ships sm_120
kernels, probes FP8 dtype support, and measures real matmul throughput
per precision so we have a hardware baseline before writing any kernels.

Usage:  .venv/bin/python verify_gpu.py
"""
import sys
import time

import torch

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


def status(ok: bool) -> str:
    return f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"


def check_device() -> tuple[int, int]:
    print(f"torch {torch.__version__}  |  built against CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        print(f"{RED}CUDA unavailable. Inside WSL check `nvidia-smi`, and confirm "
              f"the torch build is a cu12x/cu13x wheel (not the CPU wheel).{RESET}")
        sys.exit(1)

    cap = torch.cuda.get_device_capability()
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"capability  : {sm}")
    print(f"torch archs : {' '.join(torch.cuda.get_arch_list())}")

    # A wheel lacking native sm_120 SASS still runs via PTX JIT, but slowly.
    if sm not in torch.cuda.get_arch_list():
        print(f"{YELLOW}WARNING: {sm} not in this build's arch list. Kernels will be "
              f"PTX-JIT'd, which is measurably slower. Prefer a cu128+ wheel.{RESET}")
    else:
        print(f"native {sm} kernels: {status(True)}")

    return cap


def check_fp8() -> list[str]:
    """Blackwell exposes FP8 tensor cores; confirm the dtypes exist."""
    print()
    print("--- FP8 dtype support ---")
    available = []
    for name in ("float8_e4m3fn", "float8_e5m2"):
        ok = hasattr(torch, name)
        print(f"  torch.{name:16s}: {status(ok)}")
        if ok:
            available.append(name)
    return available


def bench_matmul(dtype, n: int = 4096, iters: int = 50) -> str:
    """Return sustained TFLOP/s for an n x n matmul, or why it is unsupported."""
    try:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    except Exception as exc:
        return f"{YELLOW}dtype unsupported ({type(exc).__name__}){RESET}"

    try:
        for _ in range(10):          # warm up clocks and autotuner
            _ = a @ b
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
    except Exception as exc:
        return f"{YELLOW}matmul unsupported ({type(exc).__name__}){RESET}"

    return f"{(2 * n ** 3) / elapsed / 1e12:7.1f} TFLOP/s"


def bench_fp8(n: int = 4096, iters: int = 50, fast_accum: bool = False) -> str:
    """
    FP8 GEMM throughput via torch._scaled_mm.

    Plain `a @ b` is not implemented for FP8 in PyTorch and never has been --
    FP8 matmul requires explicit per-tensor scale factors, and the right-hand
    operand must be column-major for the kernel to accept it.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        return "dtype missing"
    if not hasattr(torch, "_scaled_mm"):
        return "torch._scaled_mm missing"

    dtype = torch.float8_e4m3fn
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype)
    # .t().contiguous().t() forces column-major layout without changing values.
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dtype).t().contiguous().t()
    scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    def run():
        return torch._scaled_mm(a, b, scale_a=scale, scale_b=scale,
                                out_dtype=torch.bfloat16, use_fast_accum=fast_accum)

    try:
        for _ in range(10):
            run()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
    except Exception as exc:
        return f"{YELLOW}failed: {type(exc).__name__}: {exc}{RESET}"

    return f"{(2 * n ** 3) / elapsed / 1e12:7.1f} TFLOP/s"


def main() -> None:
    check_device()
    check_fp8()

    print()
    print("--- matmul throughput (4096^3, higher is better) ---")
    for label, dtype in [
        ("FP32", torch.float32),
        ("TF32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]:
        # Toggle TF32 so the FP32 and TF32 rows measure different things.
        torch.backends.cuda.matmul.allow_tf32 = (label == "TF32")
        print(f"  {label:18s}: {bench_matmul(dtype)}")

    # FP8 goes through a different API entirely.
    print(f"  {'FP8 e4m3':18s}: {bench_fp8(fast_accum=False)}")
    print(f"  {'FP8 e4m3 (fast)':18s}: {bench_fp8(fast_accum=True)}")

    free, total = torch.cuda.mem_get_info()
    print()
    print(f"VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print(f"{GREEN}Baseline captured. These numbers are what any custom kernel "
          f"must beat.{RESET}")


if __name__ == "__main__":
    main()
