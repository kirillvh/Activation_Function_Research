# Learnable Sine-Triangle Activations

## Starting Proposal

The initial proposal was:

```text
2*b*asin(sin(w*x/pi))/pi + (1-b)*sin(w*x/pi)
```

It has two practical problems.

1. `asin(sin(z))` has infinite autograd gradients at triangle-wave corners.
2. The inner `/pi` gives period `2*pi^2/w` and a small near-zero slope. With
   `w=1`, the activation is almost linear over ordinary normalized inputs.

The project preserves this interpretation as `sine_triangle_literal` for
controlled comparisons, but it is not recommended.

## Stable Triangle

The equivalent triangle wave is evaluated without `asin`:

```text
q = (z - pi/2) / (2*pi)
triangle(z) = 1 - 4*abs(q - round(q))
```

This keeps values in `[-1, 1]`, gives finite autograd values at corners, and
was about 40% faster than the earlier remainder-based implementation in a
CPU elementwise microbenchmark.

The corrected bounded activation is:

```text
f(x) = b*triangle(w*x) + (1-b)*sin(w*x)
```

Frequency `w` is sigmoid-constrained to a configurable positive interval and
blend `b` is sigmoid-constrained to `(0, 1)`.

## Robust Residual Variant

Bounded periodic activations replace the strong monotonic path used by common
classification networks. The better-performing variant instead learns a
periodic correction:

```text
f(x) = GELU(x) + a*(b*triangle(w*x) + (1-b)*sin(w*x))
```

The periodic amplitude `a` is also learned. It starts at `0.1` in
`gelu_sine_triangle` and `0.01` in `gelu_sine_triangle_deep`.

This design is consistent with two findings in prior work:

- [SIREN](https://arxiv.org/abs/2006.09661) shows that periodic activations
  can be effective but require careful scale and initialization.
- [Adaptive Blending Units](https://arxiv.org/abs/1806.10064) reports that
  adaptive activation scaling is important when learning combinations.

It also matches the broader lesson from
[ACON](https://arxiv.org/abs/2009.04759): useful learned activations often
preserve a robust baseline path instead of forcing all features through one
unusual nonlinearity.

## Results

All tests use paired seeds and validation-selected weights.

| Task | GELU | Hybrid | Runs |
| --- | ---: | ---: | ---: |
| MNIST CNN | 96.76 +/- 0.22% | 96.83 +/- 0.08% | 3 |
| CIFAR-10 CNN, 8,192 samples | 69.55 +/- 1.27% | 70.51 +/- 0.62% | 3 |
| CIFAR-10 CNN, full data, 120 epochs | 89.477 +/- 0.255% | 89.483 +/- 0.255% | 3 |
| Noisy PQD, 256 samples | 86.76 +/- 3.88% | 88.26 +/- 4.81% | 6 |

The bounded sine-triangle activation improved on pure sine in the signal
pilot but did not consistently beat GELU. The residual hybrid won four of six
signal seeds and two of three CIFAR seeds.

The full CIFAR-10 confirmation is the most informative result. Periodic GELU
won two paired seeds and lost one, producing a negligible `+0.007` point mean
test change while taking `1.94x` as long on CPU. Its small-data advantage did
not persist. See [PERIODIC_GELU_LONG_RUN.md](PERIODIC_GELU_LONG_RUN.md).

The deeper ResNet pilot did not establish a win. Starting amplitude `0.01`
reached 57.23% versus GELU at 56.93% on one seed, but this is too little
evidence for a claim. Starting at `0.1` slightly underperformed.

## Limitations

- The main confirmation still has only three paired seeds and one model.
- Triangle operations made the hybrid roughly 1.5-2.3 times slower than GELU.
- The learned objective remains periodic and potentially multimodal.
- More seeds, full CIFAR-10/100, accelerators, and modern large architectures
  are needed before recommending this activation in production.
- A limited literature search did not identify this exact GELU residual
  combination, but that is not proof of novelty.

Research on periodic activations is domain-dependent. BigVGAN reports strong
audio results with periodic activations and anti-aliasing, while
[periodic extrapolation tests](https://arxiv.org/abs/2209.10280) show that
periodicity alone does not guarantee robust out-of-distribution behavior.

## Commands

```powershell
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_mnist.yaml
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_cifar10.yaml
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_pqd.yaml
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_resnet18.yaml
```
