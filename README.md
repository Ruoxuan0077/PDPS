# Provable Diffusion Posterior Sampling for Bayesian Inversion

This repo contains the official implementation for the paper [Provable Diffusion Posterior Sampling for Bayesian Inversion](https://arxiv.org/abs/2512.08022)

by [Jinyuan Chang](https://sites.google.com/site/bryanchangjinyuan/), [Chenguang Duan](https://chenguangduan.github.io), [Yuling Jiao](https://jszy.whu.edu.cn/jiaoyuling/en/index.htm), Ruoxuan Li, Jerry Zhijian Yang, [Cheng Yuan](https://scholar.google.com/citations?user=UFL4YUwAAAAJ&hl=en)

--------------------

We propose a novel diffusion-based posterior sampling method within a plug-and-play (PnP) framework. Our approach constructs a probability transport from an easy-to-sample terminal distribution to the target posterior, using a warm-start strategy to initialize the particles. To approximate the posterior score, we develop a Monte Carlo estimator in which particles are generated using Langevin dynamics, avoiding the heuristic approximations commonly used in prior work. The score governing the Langevin dynamics is learned from data, enabling the model to capture rich structural features of the underlying prior distribution. 

(please add experimental results here, figures and tables)

**Methods:**
- **PDPS**: Provable Diffusion Posterior Sampling for Bayesian Inversion
- **DPS**: Diffusion Posterior Sampling (baseline)
- **TV**: Deterministic proximal-gradient reconstruction with a TV prior

---

## Structure

```
PDPS/
├── pdps.py, dps.py, tv.py          # Method-specific entry scripts
├── configs/                        # Configuration system (zero branching)
│   ├── __init__.py
│   ├── pdps.py, dps.py, tv.py
└── src/
    ├── cli.py          # Shared CLI plumbing
    ├── core/           # Execution, runtime, and run manifests
    ├── utils/          # IO and postprocessing
    ├── samplers/       # Algorithm implementations
    ├── prior/          # Unified EDM prior (VE interface)
    └── likelihood/     # Operators and likelihoods
```

---

## Pretrained Models

**EDM Prior (required for PDPS and DPS):**
```bash
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl -P data/nn/edm/
```

**Nonlinear blur model (required for nonlinear_deblur task):**
- Download: [GOPRO_wVAE.pth](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view?usp=drive_link)
- Place at: `src/likelihood/utils/bkse/experiments/pretrained/GOPRO_wVAE.pth`

---

## Quick Start

**Requirements:** Python 3.9 is the supported environment. Install the
tested dependency set in one resolver transaction:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The pinned requirements keep PyTorch 1.13, TorchMetrics, and DeepInv on
mutually compatible versions. In particular, do not install DeepInv
separately after creating the environment, because an unconstrained resolver
may upgrade PyTorch. The repository includes the small compatibility alias
needed by DeepInv 0.3.2 on PyTorch 1.13; no edit to `site-packages` is needed.
Run the commands below from the repository root.

**Paper cases and presets:**
```bash
python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097
python dps.py --paper -d ffhq -m single -t gaussian_deblur -i 097
python tv.py --paper -d ffhq -m single -t gaussian_deblur -i 097
```

**Custom experiments:**
```bash
python pdps.py -t gaussian_deblur -d ffhq -i 097 -T 0.3 --t0 0.05 -n 10
python dps.py -t gaussian_deblur -d ffhq -i 097 --scale 1.2 -n 5
python tv.py -t gaussian_deblur -d ffhq -i 097 --max-iter 50
```

For a memory-bounded batch run, choose and record an explicit chunk size:

```bash
python dps.py --paper -d ffhq -m batch -t gaussian_deblur \
  --batch-chunk-size 16
```

GPU selection follows PyTorch's logical device namespace. Use
`CUDA_VISIBLE_DEVICES` when a run should use only selected GPUs, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python dps.py --paper -d ffhq -m batch \
  -t gaussian_deblur --batch-chunk-size 16
```

**With evaluation:**
```bash
python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097 --eval
```

---

## Method Parameters

- **PDPS**: `-T` (diffusion time), `--t0` (terminal time),
  `-w` (warm-up steps)
- **DPS**: `--scale` (guidance scale), `--steps` (diffusion steps)
- **TV**: `--lambda-tv`, `--stepsize`, `--max-iter`,
  `--tv-inner-iters`, and `--tv-inner-tol`

Use `--help` for details: `python pdps.py --help`

All methods accept `--seed` and `--measurement-seed`. If `--seed` is
omitted, a seed is generated once and recorded in `run.json`; the measurement
seed defaults to 42.

Batch mode also accepts `--batch-chunk-size`. It bounds the number of images
sent through a sampler call and is recorded in `run.json` and in custom-run
fingerprints. Keep the same chunk size when reproducing a stochastic run,
because changing tensor batching can change its random-number stream. If the
option is omitted, each GPU partition is processed in one sampler call.

Add `--strict-deterministic` when byte-identical reconstructed PNGs are
required in a rerun on the same software, hardware, and GPU partition. This
opt-in mode enables fail-closed deterministic PyTorch/CUDA execution and
deterministic backward implementations for the current blur operators. It can
be slower and will raise an error if a future operator uses an unsupported
nondeterministic kernel. It does not promise bitwise equality across different
PyTorch/CUDA versions, GPU models, or GPU partitions. The `run.json` files
themselves contain timestamps and other attempt metadata, so they are not
byte-identical.

For such a rerun, explicitly pass the generated seed recorded in `run.json`
and keep the measurement seed, input files, and model weights unchanged.
Deterministic settings are process-global in PyTorch. The Python API therefore
rejects switching from a strict run to a non-strict run in the same process;
start a fresh process when switching modes. Strict library calls should use
`src.core.execute`; constructing `Runner` directly without first configuring
the deterministic runtime is rejected.

TV has no sampling randomness and requires `num_samples=1`. It uses the same
forward operators and measurements as PDPS/DPS. The implementation computes
the data gradient with PyTorch autograd and uses DeepInv only for the TV
proximal map.
The TV paper presets reuse the reported optimization parameters, while the
historical TV-only nonlinear resizing path is intentionally not reintroduced;
`--paper` therefore denotes the corrected PGD-TV baseline on the listed paper
cases, not a bit-for-bit replay of the legacy TV script.

### Reproducibility scope

The `--paper` flag selects the parameter settings and cases reported in the
paper. It does not promise byte-identical regeneration of the frozen PDPS
figures or table entries: those historical outputs predate the correction that
makes the reverse loop execute all declared `N_rev` stochastic steps, and
their original sampling seeds were not retained. The current implementation
follows the disclosed `N_rev`-step scheme. New runs record their seeds,
software and hardware environment, and GPU partition in `run.json`.

---

## Output

Paper presets retain their stable path:

`fig/{method}/paper/{mode}/{operator}/{dataset_or_image}/`

Custom runs add a 12-character configuration fingerprint so different
parameters, seeds, and sample counts cannot share a directory:

`fig/{method}/custom/{mode}/{operator}/{dataset_or_image}/{fingerprint}/`

Each new run writes an atomic `run.json` containing the effective
configuration, random seeds, Git/runtime provenance, GPU partitions, and the
exact expected output files. A nonempty target directory is rejected by
default. Use `--overwrite` to replace that exact directory, or `--resume` to
continue only when its manifest, configuration, and GPU partition agree.
Each run holds an exclusive lock next to its output directory, and PNG files
are atomically moved into place only after they are completely written. If a
machine crash leaves a lock behind, inspect the PID and hostname recorded in
the lock file before removing that stale lock manually.

Metrics: Add `--eval` to compute PSNR/SSIM and save to `metrics.txt`.
When `run.json` exists, evaluation reads only the result files declared by
that manifest; unrelated or stale PNG files are ignored.

---

## Adding New Methods

Three files required:
1. `configs/new_method.py` - Configuration
2. `src/samplers/new_method.py` - Algorithm
3. `new_method.py` - Entry script

Register in `configs/__init__.py` and `src/samplers/__init__.py`.


```
@misc{chang2025provable,
title={Provable Diffusion Posterior Sampling for {B}ayesian Inversion}, 
author={Jinyuan Chang and Chenguang Duan and Yuling Jiao and Ruoxuan Li and Jerry Zhijian Yang and Cheng Yuan},
year={2025},
note={arXiv:2512.08022},
url={https://arxiv.org/abs/2512.08022}, 
}
```

