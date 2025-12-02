# PDPS

Modular implementation of posterior sampling methods for image inverse problems.

**Methods:**
- **PDPS**: Provable Diffusion Posterior Sampling for Bayesian Inversion
- **DPS**: Diffusion Posterior Sampling (baseline)
- **PnP-Flow**: Plug-and-Play Flow (baseline)

---

## Structure

```
PDPS/
├── pdps.py, dps.py, pnp_flow.py   # Method-specific entry scripts
├── configs/                        # Configuration system (zero branching)
│   ├── __init__.py
│   ├── pdps.py, dps.py, pnp_flow.py
└── src/
    ├── core/           # Executor and runner
    ├── utils/          # IO and postprocessing
    ├── samplers/       # Algorithm implementations
    ├── prior/          # Unified EDM prior (VE interface)
    └── likelihood/     # Operators and likelihoods
```

---

## Pretrained Models

**EDM Prior (required for all methods):**
```bash
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl -P data/nn/edm/
```

**Nonlinear blur model (required for nonlinear_deblur task):**
- Download: [GOPRO_wVAE.pth](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view?usp=drive_link)
- Place at: `src/likelihood/utils/bkse/experiments/pretrained/GOPRO_wVAE.pth`

---

## Quick Start

**Requirements:**
```
torch>=1.9.0
numpy>=1.20.0
pillow>=8.0.0
pyyaml>=5.4.0
scikit-image>=0.18.0
tqdm>=4.60.0
```

**Paper reproduction:**
```bash
python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097
python dps.py --paper -d ffhq -m single -t gaussian_deblur -i 097
python pnp_flow.py --paper -d ffhq -m single -t gaussian_deblur_fft -i 097
```

**Custom experiments:**
```bash
python pdps.py -t gaussian_deblur -d ffhq -i 097 -T 0.3 -n 10
python dps.py -t gaussian_deblur -d ffhq -i 097 --scale 1.2 -n 5
python pnp_flow.py -t gaussian_deblur_fft -d ffhq -i 097 --lr-pnp 1.5 -n 10
```

**With evaluation:**
```bash
python pdps.py --paper -d ffhq -m single -t gaussian_deblur -i 097 --eval
```

---

## Method Parameters

- **PDPS**: `-T` (diffusion time), `-w` (warm-up steps)
- **DPS**: `--scale` (guidance scale), `--steps` (diffusion steps)
- **PnP-Flow**: `--lr-pnp` (learning rate)

Use `--help` for details: `python pdps.py --help`

---

## Output

Results: `fig/{method}/{paper|custom}/{mode}/{task}/{dataset}_{image_id}/`

Metrics: Add `--eval` to compute PSNR/SSIM and save to `metrics.txt`.

---

## Adding New Methods

Three files required:
1. `configs/new_method.py` - Configuration
2. `src/samplers/new_method.py` - Algorithm
3. `new_method.py` - Entry script

Register in `configs/__init__.py` and `src/samplers/__init__.py`.

