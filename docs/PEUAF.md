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

## Successive Halving

The practical signal protocol promotes four frequency starts through 5, 15,
and 25 cumulative epochs, then trains the winner to a total path length of 50
epochs.

| Seed | Fixed `w=0.5` | Successive halving |
| ---: | ---: | ---: |
| 42 | 80.59% | 84.69% |
| 43 | 85.52% | 87.60% |
| 44 | 88.01% | 89.58% |
| Mean | 84.71% | 87.29% |

A fixed-frequency staged-restart control reached 86.76%. Most of the gain came
from validation-best warm restarts; frequency selection added a smaller gain
and reduced observed variance. This distinction matters.

## Commands

```powershell
python -m activation_benchmark.expressivity_benchmark `
  --config configs/benchmark_triangle_expressivity.yaml

python -m activation_benchmark.frequency_optimization `
  --config configs/benchmark_peuaf_frequency_optimization.yaml

python -m activation_benchmark.multistart `
  --config configs/peuaf_signal_successive_halving.yaml
```
