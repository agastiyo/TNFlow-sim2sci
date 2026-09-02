# TNFlow

Companion repository for the paper: [arXiv link placeholder]

This repository contains the code and pretrained model needed to reproduce the results reported in the paper. It is not an official release of TNFlow.

TNFlow is a transformer and conditional neural spline flow that learns to invert the Shkuratov radiative transfer model for TNO surface composition. Given a reflectance spectrum, it returns a full multimodal posterior over simplex-valid compositions and grain sizes in under a second on CPU. The model is trained entirely on synthetic spectra (463,275 Shkuratov mixtures over 20 surface components) and generalizes to unseen combinations of known components. Qualitative tests on real JWST spectra reveal a sim-to-real gap that future work will address.

## Requirements

Python 3.10+ is required. CUDA is optional (CPU inference takes ~0.7 s per object).

```
pip install -r requirements.txt
pip install --no-deps cana sdoc
```

`cana` and `sdoc` must be installed with `--no-deps`.

## Data

Small data files are included in the repository:

| File | Description | Size |
|------|-------------|------|
| `data/manifest.parquet` | Metadata: spectrum IDs, train/val/OOD splits, mixture codes | 11 MB |
| `data/JWST_DiSCO-TNOs/` | Real JWST spectra for the transfer experiment (Appendix C) | 1.6 MB |
| `data/other_objects/` | Pholus, Quaoar, Iapetus spectra for inference demo | 180 KB |

The training corpus is hosted on Zenodo:

| File | Description | Size | Download |
|------|-------------|------|----------|
| `spectra.parquet` | Pre-built synthetic training corpus (463,275 spectra) | 827 MB | [Zenodo](https://doi.org/10.5281/zenodo.22240289) |

Download `spectra.parquet` and place it in `data/`:

```
data/
├── spectra.parquet        ← download from Zenodo
├── manifest.parquet
├── JWST_DiSCO-TNOs/
└── other_objects/
```

## Pre-trained Model

| Model | Epoch | Description | Size |
|-------|-------|-------------|------|
| `tnflow_checkpoint.pt` | 200 | Checkpoint used in the paper. Reproduces Table 1. | 18 MB |

The checkpoint is included in the repository.

## Reproduction

To reproduce the paper results from the pretrained checkpoint:

```
bash reproduce.sh
```

This runs two scripts in sequence:

1. **`python3.10 -m src.pipeline.test`** — Evaluates the model on the in-distribution test split (5,000 spectra) and group-holdout OOD split (374 spectra) with 500 posterior draws per query. Produces Table 1 metrics and PIT calibration figures.

2. **`python3.10 -m src.pipeline.apply_jwst`** — Applies the model to real JWST DiSCo-TNOs spectra (bowl / double-dip / cliff exemplars). Produces the appendix figure and mode report.

Both scripts use the checkpoint at `tnflow_checkpoint.pt`. Evaluation adds noise to synthetic spectra each run, so metrics may differ from the paper by small amounts within the reported standard deviations.

## Training

To train the model from scratch on the synthetic corpus:

```
python3.10 -m src.pipeline.train
```

Training requires `data/spectra.parquet` and `data/manifest.parquet`. The training loop selects the best checkpoint by validation loss and saves it to `tnflow_checkpoint.pt`. Training was performed on a single NVIDIA RTX 6000 Ada GPU (48 GB).

To rebuild the parquet corpus from raw Shkuratov model output (not required for training from the provided corpus):

```
python3.10 -m src.data_utils.store
```

## Results

Per-solution accuracy on the test and group-holdout OOD splits (500 posterior draws per query, joint composition + grain BIC-GMM mode search). Top mode is the highest-weight mode; best mode is the mode closest to the target composition in total-variation distance.

| Metric | Test (*n* = 5,000) | OOD (*n* = 374) |
|--------|-------------------|-----------------|
| Top-mode comp. TV | 0.149 ± 0.125 | 0.156 ± 0.091 |
| Best-mode comp. TV | 0.121 ± 0.098 | 0.132 ± 0.080 |
| Top-mode grain (dex) | 0.242 ± 0.212 | 0.279 ± 0.162 |
| Best-mode grain (dex) | 0.253 ± 0.201 | 0.283 ± 0.161 |
| Top mode = best (%) | 67.7 | 60.4 |
| Mean mode count | 2.50 | 2.64 |
| Multimodal (%) | 88.9 | 91.7 |
| Posterior sharpness | 0.203 ± 0.106 | 0.228 ± 0.066 |

## Source File Index

### `src/model/`

| File | Description |
|------|-------------|
| `tnflow.py` | Top-level model: tokenizer, transformer, composition CNF, and grain flow |
| `transformer.py` | Pre-norm transformer encoder with learned query pooling |
| `tokenizers.py` | Raw (λ, R) spectrum to per-channel tokens with positional encoding |
| `cnf.py` | Conditional normalizing flow over the K-component composition simplex |
| `grain.py` | Per-component grain-size posterior (shared 1-D NSF) |
| `posterior.py` | Posterior analysis: credible intervals, BIC-GMM mode search, plotting |

### `src/pipeline/`

| File | Description |
|------|-------------|
| `train.py` | Training loop: combined composition + grain NLL |
| `test.py` | Evaluation: per-solution accuracy, calibration (PIT), and timing |
| `apply_jwst.py` | Apply TNFlow to real JWST DiSCo-TNOs spectra (appendix figure + mode report) |
| `inference.py` | Interactive single-spectrum inference with plots and printed summaries |
| `checkpoint.py` | Checkpoint load / save |
| `metrics.py` | Evaluation metrics (top-N recall, TV distance, PIT, energy score) |

### `src/data_utils/`

| File | Description |
|------|-------------|
| `dataloader.py` | Builds train / val / OOD batches from SpectrumSamples with noise augmentation |
| `store.py` | Parquet-backed sample store and declarative dataset splits |
| `components.py` | Component material to integer code mapping (K=20 enum) |
| `spectrum_sample.py` | SpectrumSample dataclass: wavelength, reflectance, composition target, grains |
| `generate_shkuratov.py` | Generate synthetic spectra with the CANA Shkuratov intimate-mixture model |
| `plot_one_shkuratov.py` | Generate and plot a single Shkuratov spectrum |

### `src/data_utils/processing/`

| File | Description |
|------|-------------|
| `shkuratov.py` | Shkuratov RT model grid (bestfit files) to synthetic SpectrumSamples |
| `more_shkuratov.py` | CANA/SDOC synthetic spectra to SpectrumSamples |
| `jwst.py` | JWST DiSCo-TNOs PRISM spectra to unlabelled SpectrumSamples |
| `other_objects.py` | Miscellaneous published spectra (Pholus, Quaoar, Iapetus) to SpectrumSamples |

### `src/`

| File | Description |
|------|-------------|
| `config.py` | Project paths, device selection, and shared defaults |

## License

Code: MIT (license file will be added after the review period)

Data (`spectra.parquet`, `manifest.parquet`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
