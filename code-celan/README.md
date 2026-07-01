# AdTrans: Transformer-Based Cathode Voltage Prediction

This repository contains the code used to train and evaluate **AdTrans**, a dual-stream Transformer model for predicting average discharge voltages of cathode materials. The pipeline covers descriptor generation, CHGNet fine-tuning, hyperparameter optimization (HPO), model training, explainable AI (XAI) analysis, and an MLP baseline.

Pre-computed descriptor CSV files are bundled under `AdTrans/descriptors/`, so model training can be reproduced without regenerating features from scratch. **Raw electrode records and element lookup tables are not included in this repository** due to size limits; download them from Zenodo (see [Source Data](#source-data)).

All paths below are relative to the repository root (`proj/`).

---

## Repository Structure

```
proj/
├── data_processing/              # Raw data → descriptor CSVs, CHGNet fine-tuning
│   ├── source_data/              # Not bundled — download from Zenodo (see below)
│   │   ├── split/                # train.csv, val.csv, test.csv
│   │   ├── magpie.csv
│   │   ├── mat2vec.csv
│   │   └── skipatom.csv
│   ├── descriptors_generator.py  # Element, property, and pretrained CHGNet descriptors
│   ├── dataset_split.py          # Split utilities
│   └── chgnet_finetune_dual/     # Voltage-aware CHGNet fine-tuning + export
├── AdTrans/                      # Main Transformer model
│   ├── main.py                   # Entry point: HPO, training, evaluation, XAI
│   ├── descriptors/              # Descriptor CSVs consumed by the model
│   ├── data/                     # Preprocessing and structure token placement
│   ├── models/                   # Transformer architecture
│   ├── training/                 # Trainer, evaluator, Optuna HPO, pruning pipeline
│   ├── xai/                      # Integrated Gradients and feature ablation
│   └── utils/                    # Configuration and plotting utilities
└── baseline_mlp/                 # Flattened MLP baseline on the same descriptors
    └── main.py
```

---

## Source Data

Raw source files are hosted on Zenodo to keep the code repository lightweight:

**[source data for AdTrans](https://doi.org/10.5281/zenodo.21095579)**  
DOI: [`10.5281/zenodo.21095579`](https://doi.org/10.5281/zenodo.21095579) · License: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

| File | Place under |
|------|-------------|
| `train.csv`, `val.csv`, `test.csv` | `data_processing/source_data/split/` |
| `magpie.csv`, `mat2vec.csv`, `skipatom.csv` | `data_processing/source_data/` |

After download, the layout should be:

```
data_processing/source_data/
├── split/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── magpie.csv
├── mat2vec.csv
└── skipatom.csv
```

These files are required only for **descriptor regeneration** or **CHGNet fine-tuning**. The bundled descriptors in `AdTrans/descriptors/` are sufficient for model training via [Quick Start](#quick-start-reproduce-main-results).

The Zenodo record documents provenance of the compiled dataset (Mat2vec, Skipatom, Magpie embeddings and Materials Project–derived cathode records). See the dataset description on Zenodo for full references.

---

## Dataset

The default split contains **6,025** cathode records (train / val / test = 4,814 / 607 / 604):

| Split | Samples |
|-------|---------|
| Train | 4,814   |
| Val   | 607     |
| Test  | 604     |

Each record includes a discharge formula, average voltage label, and (where applicable) charge/discharge crystal structures serialized as JSON. Descriptor CSVs store flattened feature vectors aligned by `battery_id`.

---

## Requirements

Tested with **Python 3.10** and a CUDA-capable GPU for training. Core dependencies:

| Package | Purpose |
|---------|---------|
| PyTorch | Model training and inference |
| NumPy, pandas | Data handling |
| scikit-learn | Metrics and preprocessing |
| pymatgen | Composition and structure parsing |
| CHGNet | Structure embeddings |
| Optuna, plotly, kaleido | Hyperparameter search and Optuna plots |
| matplotlib, seaborn | Result and XAI figures |

Example environment setup:

```bash
conda create -n papernet python=3.10 -y
conda activate papernet

pip install torch numpy pandas scikit-learn pymatgen chgnet optuna matplotlib seaborn plotly kaleido
```

> **Note:** Set `TRAINING_CONFIG['device']` in `AdTrans/utils/config.py` (and the baseline config) to `'cpu'` if no GPU is available. Training will be substantially slower.

---

## Quick Start (Reproduce Main Results)

Pre-generated descriptors are already present in `AdTrans/descriptors/`. Run training from the module directories (relative paths assume the working directory shown below).

### 1. Train AdTrans

```bash
cd AdTrans
python main.py
```

This executes, in order:

1. Load Skipatom (encoder) and Magpie (decoder) descriptors plus finetuned CHGNet structure embeddings (`chgnet_ft_v_dual`, 128-d).
2. Optuna HPO (**100 trials** by default; see Configuration).
3. Early-stopped training on the best hyperparameters.
4. Test-set evaluation and artifact export.
5. XAI analysis (Integrated Gradients, feature ablation, attention maps).

**Outputs** are written to `AdTrans/results/skipatom_magpie/`, including:

| Artifact | Description |
|----------|-------------|
| `model.pth` | Trained model weights |
| `best_hyperparameters.json` | Optuna best trial |
| `data_stats.npy` | Standardization statistics |
| `predictions_skipatom_magpie_*.csv` | Per-split predictions |
| `plots/` | Parity plots, attention heatmaps, XAI figures |
| `*_importance_*.csv` | Ranked feature / token importance |

Terminal output reports MAE, RMSE, and R² on train, validation, and test splits.

### 2. Train MLP Baseline

```bash
cd baseline_mlp
python main.py
```

The baseline reads the same descriptor CSVs from `AdTrans/descriptors/`, flattens enabled input blocks (encoder / decoder / structure), and runs an independent Optuna HPO loop. Results are saved under `baseline_mlp/results/`.

---

## Full Data Pipeline (Optional)

Regenerate descriptors only if you modify the raw dataset or split assignment. **Download the source files from Zenodo first** ([Source Data](#source-data)).

### Step 0 — Download and place source files

Download all six CSV files from [https://doi.org/10.5281/zenodo.21095579](https://doi.org/10.5281/zenodo.21095579) and copy them into `data_processing/source_data/` as shown above (~75 MB total).

### Step 1 — Generate element, property, and pretrained CHGNet descriptors

```bash
cd data_processing
python descriptors_generator.py
```

Control generation tasks via `GENERATION_CONFIG` at the top of `descriptors_generator.py` (default: Skipatom, mat2vec, Magpie element vectors; Magpie property tokens including `band_gap` and `energy_above_hull`; 128-d charge|discharge CHGNet embeddings from the pretrained model).

Output directory: `data_processing/descriptor/`.

### Step 2 — Fine-tune CHGNet on voltage labels

```bash
cd data_processing
python -m chgnet_finetune_dual.train_voltage_repr
python -m chgnet_finetune_dual.export_descriptors
```

Settings are defined in `chgnet_finetune_dual/config.py`. The fine-tuned backbone is saved to `chgnet_finetune_dual/results/voltage_repr/best_backbone.pt`. Exported structure descriptors are written to `data_processing/descriptors/structure_base/chgnet_ft_v_dual/`.

### Step 3 — Copy descriptors into AdTrans

AdTrans reads features only from `AdTrans/descriptors/`. After regeneration, copy the relevant subfolders:

```bash
# Linux / macOS example
cp -r data_processing/descriptor/element_base/*   AdTrans/descriptors/element_base/
cp -r data_processing/descriptor/property_base/*  AdTrans/descriptors/property_base/
cp -r data_processing/descriptors/structure_base/chgnet_ft_v_dual \
      AdTrans/descriptors/structure_base/
```

On Windows, use `xcopy /E /I /Y` or equivalent.

Then proceed with **Quick Start** above.

---

## Default Model Configuration

Key settings in `AdTrans/utils/config.py` (paper defaults):

| Setting | Value | Description |
|---------|-------|-------------|
| `descriptor_pair` | `('skipatom', 'magpie')` | Encoder / decoder descriptor sources |
| `structure_descriptor_name` | `chgnet_ft_v_dual` | Finetuned dual-state CHGNet embeddings |
| `structure_placement` | `'all'` | Structure tokens in both encoder and decoder |
| `structure_descriptor_dim` | `128` | Concatenated charge + discharge embedding |
| `structure_dual_token_split` | `False` | Single 128-d structure token (not split into 2×64) |
| `OPTIMIZER_CONFIG['n_trials']` | `100` | Optuna trials per HPO stage |
| `OPTIMIZER_CONFIG['seed']` | `42` | Random seed for reproducibility |

### Token pruning (optional)

Set `PRUNING_STAGE_CONFIG['enabled'] = True` in `AdTrans/utils/pruning_config.py` to run the progressive decoder-token pruning pipeline instead of a single training run. Pruned results are saved under `AdTrans/results/pruning/`.

### MLP input ablation

Toggle input blocks in `baseline_mlp/config.py`:

```python
INPUT_CONFIG = {
    "use_encoder": True,
    "use_decoder": True,
    "use_structure": True,
}
```

---

## Reproducibility Notes

- Global seed: **42** (`DATASET_CONFIG` / `OPTIMIZER_CONFIG` / `PRUNING_STAGE_CONFIG`).
- HPO uses Optuna with fixed seed; full bitwise reproducibility across hardware may vary slightly with CUDA.
- For faster debugging, reduce `OPTIMIZER_CONFIG['n_trials']` (e.g., to 5) and disable XAI in `XAI_CONFIG` or `utils/visualization_config.py`.

---

## Troubleshooting

| Issue | Suggested fix |
|-------|----------------|
| Missing `source_data/` CSVs | Download from [Zenodo](https://doi.org/10.5281/zenodo.21095579) and place files per [Source Data](#source-data) |
| `FileNotFoundError` for descriptor CSVs | Run `main.py` from `AdTrans/`; verify `descriptors/structure_base/chgnet_ft_v_dual/train.csv` exists |
| CUDA out of memory | Lower `batch_sizes` in config or set `device='cpu'` |
| Optuna plot export fails | Install `plotly` and `kaleido`, or set `OPTIMIZER_CONFIG['visualization']['enabled'] = False` |
| MLP structure path error | Align `structure_descriptor_name` in `baseline_mlp/config.py` with AdTrans |
| CHGNet export fails | Run `train_voltage_repr` first; confirm `best_backbone.pt` exists |

---

## Citation

If you use this code, please cite the associated publication:

```bibtex
@article{your_paper_key,
  title   = {Your Paper Title},
  author  = {Author List},
  journal = {Journal of Energy Chemistry},
  year    = {2026},
  note    = {Update with final bibliographic details upon acceptance}
}
```

If you use the raw source data, please also cite the Zenodo record:

```bibtex
@dataset{zhang2026adtrans_source,
  author       = {Zhang, Xingshou and Chen, Haiyuan and Niu, Xiaobin},
  title        = {source data for AdTrans},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21095579},
  url          = {https://doi.org/10.5281/zenodo.21095579}
}
```

---

## License

Specify your license here (e.g., MIT, Apache-2.0, or institutional terms) before public release.
