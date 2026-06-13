# Periodic Activation Benchmark

A reproducible PyTorch lab for training, plotting, and comparing activation
functions on MNIST, CIFAR-10/100, and noisy 1D power-quality signals.

The repository began as a PEUAF investigation and now includes a learnable
sine-triangle family. 

So far there are two modest but interesting results:
1. Using an evolutionary search to optimize the frequency paramater of PEUAF and then
following up with a conventional backpropgation based optimization, was found to improve
the overall performance of PEUAF because its learnable frequency paramater tends to have
local minima which are difficult for a gradient based optimizer to escape so a global search 
warm up phase is beneficial.
with basins
2. a GELU-based periodic residual improved mean accuracy in two small research protocols,
was neutral on MNIST, and did not establish a reliable advantage in a deeper
ResNet.

## Research Summary

All values use validation-selected weights, fixed data splits, and paired
initialization seeds. CIFAR-10 uses only 8,192 training examples, so these are
diagnostic experiments rather than leaderboard results.

| Task | Runs | GELU | GELU + sine-triangle | Change |
| --- | ---: | ---: | ---: | ---: |
| MNIST CNN | 3 | 96.76 +/- 0.22% | 96.83 +/- 0.08% | +0.07 |
| CIFAR-10 CNN | 3 | 69.55 +/- 1.27% | 70.51 +/- 0.62% | +0.96 |
| Noisy low-data PQD signals | 6 | 86.76 +/- 3.88% | 88.26 +/- 4.81% | +1.50 |

The deeper CIFAR ResNet pilot was inconclusive. GELU reached 56.93% on the
tested seed. The hybrid reached 56.54% with periodic amplitude `0.1` and
57.23% with amplitude `0.01`. Periodic variants were roughly 1.5-2.3 times
slower than GELU in these CPU runs.

Initially I tried to improve the performance of PEUAF by blending between its triangle part and a smooth sine wave but the benchmarks were dissapointing.
However the following periodic modification of GELU(x) was found to be promising on the (so far) limited tests of this repository:

```text
GELU(x) + a * (b * triangle(w*x) + (1-b) * sin(w*x))
```

`a`, `b`, and `w` are learnable and constrained. See
[docs/SINE_TRIANGLE.md](docs/SINE_TRIANGLE.md) for the derivation, numerical
pitfalls, results, and limitations.

The PEUAF work found a different useful result: periodic expressivity creates
a highly multimodal frequency objective. A custom frequency-only population
search followed by backpropagation is more practical than evolving every
network weight. See [docs/PEUAF.md](docs/PEUAF.md).

## Features

- YAML-configured train, validation, test, and parameter sweeps
- MNIST, CIFAR-10, CIFAR-100, compact/deep CNNs, ResNet-18, and 1D signal CNN
- Augmentation, deterministic subsets, paired seeds, and best-validation tests
- TensorBoard, CSV/JSON histories, and benchmark PNGs
- Step-numbered checkpoints with configurable latest-N retention
- Checkpoint resume, warm starts, and standalone evaluation
- Activation value/derivative plots with CSV and metadata export
- PEUAF grid, multi-start, PyGAD/TorchGA, evolution, and successive halving
- Configurable PyTorch thread counts and Windows process affinity
- Double-clickable Windows launchers and matching Linux Bash scripts

## Setup

Python 3.10 or newer:

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Development and optional evolutionary dependencies:

```text
python -m pip install -r requirements-dev.txt
```

## Download Datasets

Training configs can download image datasets automatically because
`data.download` defaults to `true`. To prepare every dataset before training,
use the explicit downloader:

Windows:

```powershell
download_datasets.bat
```

Linux:

```bash
bash download_datasets.sh
```

The downloader uses TorchVision's official download, integrity-check, and
archive-extraction implementation. It prepares both train and test splits:

```text
data/MNIST/                 MNIST raw and processed files
data/cifar-10-batches-py/   extracted CIFAR-10 files
data/cifar-100-python/      extracted CIFAR-100 files
```

Download only selected datasets or use another root:

```text
python -m activation_benchmark.download_datasets \
  --datasets mnist cifar10 --root data
```

Accepted names are `mnist`, `cifar10`, `cifar100`, and `all`. Downloads are
idempotent: existing verified files are reused. The synthetic PQD dataset is
generated locally at runtime and has nothing to download. For offline
training after preparation, set `data.download: false` in the YAML config or
pass `--set data.download=false`.

## Quick Start

Double-click `train.bat`, run `bash train.sh` on Linux, or invoke Python
directly:

```powershell
python -m activation_benchmark.train --config configs/mnist.yaml
```

Run a smoke test:

```powershell
python -m activation_benchmark.train --config configs/smoke.yaml
python -m pytest
```

Override any YAML value:

```powershell
python -m activation_benchmark.train --config configs/cifar10.yaml `
  --set model.activation=gelu_sine_triangle training.epochs=20
```

Run a generic benchmark:

```powershell
python -m activation_benchmark.benchmark `
  --config configs/benchmark_activations.yaml
```

Research comparisons:

```powershell
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_mnist.yaml
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_cifar10.yaml
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_pqd.yaml
```

The ResNet config uses the depth-safer `gelu_sine_triangle_deep` variant,
whose periodic amplitude starts at `0.01`:

```powershell
python -m activation_benchmark.benchmark `
  --config configs/benchmark_sine_triangle_resnet18.yaml
```

## Activation Plots

Double-click `plot_activations.bat`, or:

```powershell
python -m activation_benchmark.plot_activations `
  --activations peuaf sine_triangle gelu gelu_sine_triangle `
  --w 0.5 1.0 --blend 0.25 0.5 0.75
```

Outputs include `activation_shapes.png`, sampled values and derivatives in
CSV, and JSON metadata.

## PEUAF Optimization

Run the one-parameter basin study:

```powershell
python -m activation_benchmark.frequency_optimization `
  --config configs/benchmark_peuaf_frequency_optimization.yaml
```

Run frequency-only evolution or successive halving before final
backpropagation:

```powershell
python -m activation_benchmark.multistart `
  --config configs/peuaf_evolutionary_multistart.yaml
python -m activation_benchmark.multistart `
  --config configs/peuaf_signal_successive_halving.yaml
```

For one scalar frequency, a grid is simpler and more reliable than a genetic
algorithm. The evolutionary workflow becomes more relevant when several
activation frequencies must be searched jointly.

## Checkpoints And TensorBoard

Checkpoints are named `step_00000123.pt`. `checkpoint.keep_latest` controls
retention and defaults to three. Each checkpoint also stores the
validation-best model weights.

Resume an existing run:

```powershell
python -m activation_benchmark.train --config configs/mnist.yaml `
  --set experiment.name=my_run checkpoint.resume=latest
```

Launch TensorBoard with `tensorboard.bat`, or:

```powershell
python -m tensorboard.main --logdir runs --port 6006
```

On Linux, use `bash tensorboard.sh`.

## Platform Launchers

Windows `.bat` and Linux `.sh` launchers provide the same commands:

| Purpose | Windows | Linux |
| --- | --- | --- |
| Download datasets | `download_datasets.bat` | `bash download_datasets.sh` |
| Train MNIST | `train.bat` | `bash train.sh` |
| Train CIFAR-10 | `train_cifar10.bat` | `bash train_cifar10.sh` |
| Train CIFAR ResNet-18 | `train_cifar10_resnet18.bat` | `bash train_cifar10_resnet18.sh` |
| Train synthetic PQD | `train_synthetic_pqd.bat` | `bash train_synthetic_pqd.sh` |
| PEUAF multi-start | `train_peuaf_multistart.bat` | `bash train_peuaf_multistart.sh` |
| Generic benchmark | `benchmark.bat` | `bash benchmark.sh` |
| CIFAR-10 benchmark | `benchmark_cifar10.bat` | `bash benchmark_cifar10.sh` |
| Sine-triangle benchmark | `benchmark_sine_triangle.bat` | `bash benchmark_sine_triangle.sh` |
| Plot activations | `plot_activations.bat` | `bash plot_activations.sh` |
| TensorBoard | `tensorboard.bat` | `bash tensorboard.sh` |

Arguments are forwarded to the Python command. The Bash scripts use
`.venv/bin/python` when available, otherwise `python3`. Override that choice
with, for example, `PYTHON=python3.12 bash train.sh`.

The checked-in research configs contain CPU affinity tuned for the original
Windows workstation. Bash training and benchmark launchers disable that
machine-specific affinity by default so they work on Linux systems with
different CPU counts. Set
`ACTIVATION_BENCHMARK_DISABLE_AFFINITY=0 bash train_cifar10.sh` to use the
affinity from the YAML config.

## Project Layout

```text
activation_benchmark/  training, models, activations, plots, and research tools
configs/               reusable training and benchmark protocols
docs/                  concise research reports
tests/                 unit and smoke coverage
runs/                  generated outputs, ignored by Git
data/                  downloaded datasets, ignored by Git
```

## References

- [EUAF and PEUAF paper](https://arxiv.org/abs/2407.09580)
- [SIREN](https://arxiv.org/abs/2006.09661)
- [Adaptive Blending Units](https://arxiv.org/abs/1806.10064)
- [ACON learnable activations](https://arxiv.org/abs/2009.04759)
- [BigVGAN periodic activations](https://arxiv.org/abs/2206.04658)
- [Periodic activations and extrapolation](https://arxiv.org/abs/2209.10280)

## License

MIT
