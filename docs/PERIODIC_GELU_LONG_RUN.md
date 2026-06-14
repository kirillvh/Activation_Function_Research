# Full CIFAR-10 Periodic GELU Study

## Question

Does the small-data advantage previously observed for

```text
GELU(x) + a * (b * triangle(w*x) + (1-b) * sin(w*x))
```

survive a substantially longer, full-data CIFAR-10 comparison?

## Protocol

- Date: June 14, 2026
- Dataset: CIFAR-10
- Split: 45,000 train, 5,000 validation, 10,000 test
- Architecture: project standard CNN, 550,570 baseline parameters
- Epochs: 120
- Optimizer: AdamW, learning rate `0.001`, weight decay `0.0001`
- Schedule: cosine decay to zero
- Augmentation: random crop, horizontal flip, random erasing
- Seeds: 42, 43, and 44, paired between activations
- Selection: best validation checkpoint evaluated once on the test set
- Hardware: CPU-only i9-12900K class system, eight P cores with SMT

The periodic GELU formula and initialization were not changed for this study.

## Results

| Seed | GELU test | Periodic GELU test | Paired change |
| ---: | ---: | ---: | ---: |
| 42 | 89.82% | 89.14% | -0.68 |
| 43 | 89.21% | 89.56% | +0.35 |
| 44 | 89.40% | 89.75% | +0.35 |
| Mean | 89.477% | 89.483% | +0.007 |

Population standard deviation was `0.255` points for both methods. With only
three pairs, the approximate 95% confidence interval for the paired mean
difference is `[-1.47, +1.48]` points.

The paired standard deviation was `0.595` points. As a rough planning
calculation at 80% power and a two-sided 5% significance level, detecting a
true `0.50`-point effect would require about 12 paired seeds; detecting a
`0.25`-point effect would require about 45. These estimates are approximate
and potentially optimistic because the variance itself was estimated from
only three pairs. The current study can reject a large, consistent advantage,
but it cannot resolve a small one.

Best validation accuracy was slightly higher for periodic GELU:
`90.553 +/- 0.096%` versus `90.340 +/- 0.185%`. Final-model test accuracy was
also slightly higher, `89.587%` versus `89.503%`, but neither difference is
large enough to establish an advantage.

Periodic GELU averaged 111.20 minutes per run versus 57.42 minutes for GELU,
or `1.94x` the CPU training time.

## Interpretation

This study does not support the earlier hypothesis that periodic GELU
generally improves CIFAR-10 classification. The small 8,192-example pilot
reported `+0.96` points, but the full-data, 120-epoch result is effectively
zero. The pilot was therefore likely dominated by small-sample or
optimization variance.

Repeating only a few more seeds would still leave a wide interval. A useful
replication should target at least 12 paired seeds if a half-point improvement
is considered practically meaningful.

The learned parameters moved consistently across all three seeds. Mean
frequency converged near `0.35`, triangle blend near `0.68`, and periodic
amplitude near `0.143`. The optimizer used the periodic branch, but its extra
flexibility did not improve generalization enough to offset its cost.

This is a result about this architecture and protocol, not proof that
periodic residuals are useless in every domain. Audio, implicit
representations, and other tasks with genuinely periodic structure remain
better-motivated targets.

## Artifacts

- [Aggregate CSV](results/periodic_gelu_cifar10_full_120epoch/aggregate.csv)
- [Per-run CSV](results/periodic_gelu_cifar10_full_120epoch/runs.csv)
- [Paired accuracy](images/periodic_gelu_cifar10_full_120epoch/paired_accuracy.png)
- [Learning curves](images/periodic_gelu_cifar10_full_120epoch/learning_curves.png)
- [Learned parameters](images/periodic_gelu_cifar10_full_120epoch/activation_parameters.png)
- Reproduction config: `configs/benchmark_periodic_gelu_cifar10_long.yaml`
