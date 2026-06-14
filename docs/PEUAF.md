# PEUAF: Expressivity Versus Optimization

The implementation follows the formula in
[Don't Fear Peculiar Activation Functions: EUAF and Beyond](https://arxiv.org/abs/2407.09580):

```text
PEUAF(x) =
  abs(w*x - 2*floor((w*x + 1)/2))  when x >= 0
  x/(1 + abs(x))                    when x < 0
```

The bracket with short horizontal feet in the paper is `floor`, not an
absolute value. This notation is easy to misread and is the one formula pitfall
worth highlighting.

## Main Finding

A single PEUAF neuron fits the periodic triangle target essentially exactly:

| One-neuron model | Test MSE |
| --- | ---: |
| PEUAF | `3.35e-14` |
| Tanh | `0.0781` |
| ReLU | `0.0936` |

That representational result does not imply that gradient descent can find
the useful frequency. For target frequency `0.37`, initialization at `0.4`
converged while initialization at `0.5` remained in another basin.

## Batch Size Does Not Search Frequencies

A larger batch estimates the gradient at the current frequency more
accurately; it does not evaluate more candidate frequencies. In the bad-basin
study, noise from small batches occasionally escaped:

| Batch size | Exact-basin success |
| ---: | ---: |
| 1 | 70% |
| 8 | 30% |
| 32 | 10% |
| 128 | 0% |
| 512 | 0% |

Small batches are therefore an unreliable exploration mechanism, not a global
optimizer.

## Search Before Backpropagation

Each initializer below received the same full-batch Adam refinement:

| Initializer | Success |
| --- | ---: |
| Gradient descent from `w=0.5` | 0/5 |
| 24-point random search | 4/5 |
| 24 vectorized Adam starts | 5/5 |
| 201-point grid | 5/5 |
| PyGAD | 5/5 |
| TorchGA | 5/5 |

For one scalar, use the grid. The project contribution is a selective hybrid
for larger models: evolve or race only activation frequencies, let ordinary
weights use backpropagation, then warm-start final training from the best
validation candidate.

TorchGA flattens all model parameters into one genetic vector. That is useful
for tiny models but spends most of a CIFAR population budget evolving
convolution weights that gradient methods already handle well.

## Twelve-Seed Synthetic-Signal Confirmation

A larger confirmation study separated uninterrupted PEUAF, a restart-matched
control, and true two-generation frequency evolution on noisy low-data
power-quality signals:

| Condition | Test accuracy | CPU time/seed |
| --- | ---: | ---: |
| GELU | 91.85 +/- 3.37% | 14.82 s |
| Direct PEUAF | 84.65 +/- 3.18% | 31.81 s |
| Staged PEUAF | 83.52 +/- 3.59% | 32.39 s |
| Evolved PEUAF | 86.91 +/- 3.61% | 66.16 s |

Evolution improved over the staged control by `+3.39` points with a paired
95% interval of `[+0.55, +6.23]`. It improved over direct PEUAF by `+2.26`
points, although that interval still crossed zero. Evolved PEUAF remained
`4.94` points behind GELU.

Direct PEUAF initialized at `w=0.5` finished near `0.485`. Evolution selected
starts averaging `0.706`, and subsequent backpropagation changed them by only
`-0.001` on average. This supports the basin hypothesis directly: gradient
descent refines the selected basin but rarely discovers a distant one.

See [PEUAF_EVOLUTION_CONFIRMATION.md](PEUAF_EVOLUTION_CONFIRMATION.md).

## Full CIFAR-10 Confirmation

The synthetic-signal gain did not transfer to the exact full CIFAR-10
protocol used for the Periodic GELU study:

| Condition | Test accuracy | CPU time/seed |
| --- | ---: | ---: |
| GELU | 89.477 +/- 0.255% | 57.42 min |
| Periodic GELU | 89.483 +/- 0.255% | 111.20 min |
| Direct PEUAF | 87.567 +/- 0.019% | 122.22 min |
| Evolved PEUAF | 86.870 +/- 0.273% | 161.21 min |

Direct PEUAF trailed GELU by `1.91` points with a paired 95% interval of
`[-2.74, -1.08]`. Evolution selected `w=0.77` in all three seeds but finished
`0.70` points below direct PEUAF on average. The search was consistent at
finding the best five-epoch candidate, but that early ranking did not produce
the best long-run model.

This negative transfer is scientifically useful: PEUAF's optimization basin
problem is real, but searching the initial frequency is not a universal
remedy. See [PEUAF_CIFAR10_CONFIRMATION.md](PEUAF_CIFAR10_CONFIRMATION.md).

## Earlier Successive-Halving Pilot

The practical signal protocol promotes four frequency starts through 5, 15,
and 25 cumulative epochs, then trains the winner to a total path length of 50
epochs.

| Seed | Fixed `w=0.5` | Successive halving |
| ---: | ---: | ---: |
| 42 | 80.59% | 84.69% |
| 43 | 85.52% | 87.60% |
| 44 | 88.01% | 89.58% |
| Mean | 84.71% | 87.29% |

This three-seed pilot suggested a gain but did not cleanly isolate restarts.
The larger confirmation above supersedes it and shows the opposite: the
restart-only control did not help, while frequency selection did.

## Commands

```powershell
python -m activation_benchmark.expressivity_benchmark `
  --config configs/benchmark_triangle_expressivity.yaml

python -m activation_benchmark.frequency_optimization `
  --config configs/benchmark_peuaf_frequency_optimization.yaml

python -m activation_benchmark.multistart `
  --config configs/peuaf_signal_successive_halving.yaml

python -m activation_benchmark.peuaf_search_benchmark `
  --config configs/benchmark_peuaf_evolution_confirmation.yaml

python -m activation_benchmark.cifar_peuaf_benchmark `
  --config configs/benchmark_peuaf_cifar10_confirmation.yaml
```
