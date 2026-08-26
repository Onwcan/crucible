# Ablation results

Control (`gqa/rope/swiglu/pre-rmsnorm`): **4.3733** ± 0.0157 over 3 seeds, 24.8% MFU

| Axis | Variant | Val loss | SD | Δ vs control | MFU | Verdict |
|---|---|---:|---:|---:|---:|---|
| `attention` | `mha` | 4.3702 | 0.0131 | -0.0031 | 29.2% | within noise |
| `attention` | `mqa` | 4.3663 | 0.0107 | -0.0070 | 24.8% | within noise |
| `pos_encoding` | `alibi` | 4.3545 | 0.0195 | -0.0188 | 21.2% | within noise |
| `pos_encoding` | `learned` | 4.5576 | 0.0102 | +0.1843 | 25.3% | worse |
| `pos_encoding` | `none` | 4.6312 | 0.0159 | +0.2579 | 24.8% | worse |
| `activation` | `gelu` | 4.3728 | 0.0073 | -0.0005 | 24.5% | within noise |
| `norm` | `layernorm` | 4.3684 | 0.0022 | -0.0049 | 9.4% | within noise |
| `norm_placement` | `post` | 4.5554 (2/3) | 0.0347 | +0.1821 | 24.1% | UNSTABLE (1/3 diverged) |

**Verdict rule.** A seed whose best validation loss exceeds the control by more than 1.0 is classified as *diverged* and excluded from the mean and SD — averaging a failed run in would inflate the SD, and the inflated SD would then mask the very difference it should reveal. Any variant with a diverged seed is reported as `UNSTABLE` regardless of how the surviving seeds performed.

For the remainder, |Δ| must exceed 2× the larger seed standard deviation (2σ = 0.0313 for the control) to count as a difference. Anything inside that band is `within noise`, not a finding.
