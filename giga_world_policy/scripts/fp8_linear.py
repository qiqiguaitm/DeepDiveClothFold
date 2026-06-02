"""Native FP8 (e4m3) Linear using torch._scaled_mm — uses Blackwell FP8 tensor cores
directly, so it does NOT need torchao's cpp extensions (which require torch>=2.11).

Rowwise dynamic quantization: per-output-channel weight scale + per-token activation
scale. This is the high-accuracy fp8 recipe (near-lossless: each row keeps its own
dynamic range), and it halves weight bytes read per GEMM — the win in the bs=1
weight-bandwidth-bound regime. Activations are quantized on the fly (fused by
torch.compile), so the matmul runs on hardware fp8 tensor cores.
"""
import torch
import torch.nn as nn

FP8_MAX = 448.0  # e4m3 max representable magnitude


def _quant_weight_rowwise(w):  # w: [N, K] bf16/fp32
    w = w.float()
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [N,1]
    scale = amax / FP8_MAX
    wq = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)  # [N,K]
    return wq, scale.squeeze(1).float()  # scale: [N]


class FP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        N, K = lin.weight.shape
        wq, scale = _quant_weight_rowwise(lin.weight.data)
        self.register_buffer("wq", wq)                 # [N,K] fp8
        self.register_buffer("w_scale", scale.view(1, N))  # [1,N] f32
        self.bias = nn.Parameter(lin.bias.data.clone()) if lin.bias is not None else None
        self.N, self.K = N, K

    def forward(self, x):
        orig = x.shape
        x2 = x.reshape(-1, self.K)
        amax = x2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [M,1]
        x_scale = (amax / FP8_MAX).float()
        xq = (x2 / x_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            xq, self.wq.t(),               # [M,K] @ [K,N] (col-major)
            scale_a=x_scale,               # [M,1] per-token
            scale_b=self.w_scale,          # [1,N] per-channel
            bias=self.bias.to(x.dtype) if self.bias is not None else None,
            out_dtype=x.dtype,
        )
        return out.reshape(*orig[:-1], self.N)


def swap_linears_to_fp8(module, min_k=256):
    """Recursively replace nn.Linear (with K,N divisible by 16) by FP8Linear."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            N, K = child.weight.shape
            if K >= min_k and K % 16 == 0 and N % 16 == 0:
                setattr(module, name, FP8Linear(child).to(child.weight.device))
                n += 1
        else:
            n += swap_linears_to_fp8(child, min_k)
    return n
