# battery-swelling-prediction

Battery swelling / thickness prediction workflows built around two complementary data sources:

- **workbook-style battery test datasets**: cycle, thickness, DCIR, OCV, ACIR, and EIS measurements
- **raw device / tester exports**: high-resolution `current / voltage / time` signals used for time-domain feature extraction

This repo supports three main tasks:

1. **frequency-domain ECM fitting** from EIS data
2. **time-domain ECM-inspired feature extraction** from raw device data
3. **swelling / thickness prediction** with classical ML, neural networks, and Transformers

## What This Repo Is For

At a high level, the project answers:

- how well lab-side EIS-derived ECM features track battery swelling
- whether raw device-side time-domain signals can provide deployable ECM-like features
- how those features perform in downstream thickness prediction models

There are two important modeling settings in this repo:

- **Hybrid / research setting**: lab-side frequency-domain ECM features and device-side time-domain features can both be used
- **Device-oriented setting**: only device-observable features are used as model inputs; frequency-domain ECM is used only offline to help define priors / bounds

## Data Types

This README intentionally uses **generic dataset names** rather than any one local folder layout.

### 1) Workbook dataset

This is the structured Excel-style dataset that usually contains:

- thickness labels
- cycle-level summaries
- DCIR / ACIR / OCV
- EIS sheets such as `02_PreEIS`, `03-4_EIS`, `04_PostEIS`

In commands below, this is usually referenced as:

```text
./dataset/udc_xlsx
```

### 2) Raw device dataset

This is the raw tester / Maccor-style dataset that usually contains:

- `current`
- `voltage`
- `capacity`
- `energy`
- `test time`
- `cycle` / step-level information

In commands below, this is usually referenced as:

```text
./dataset/raw_data
```

### Why Both Matter

- The **workbook dataset** provides the swelling / thickness labels used for supervised training.
- The **raw dataset** provides the high-resolution time-domain signal needed for device-side ECM fitting.

So the usual training flow is:

```text
workbook dataset -> labels + cycle/DCIR/capacity features
raw dataset      -> time-domain ECM-inspired features
merge by serial/cycle -> model training
```

## Main Workflows

### A. Frequency-Domain ECM Workflow

Use workbook EIS sheets to fit ECM parameters with `src/ecm_fit.py`, then summarize those results into ML-ready features or time-domain priors.

### B. Time-Domain ECM Workflow

Use raw `current / voltage / time` data to extract device-side ECM-inspired features with `src/extract_device_ecm_features.py`.

The simplified time-domain ECM used in this branch is:

```text
Vocv - R0 - (Rsei || Csei) - Rct - (Rw1 || Cw1) - (Rw2 || Cw2)
```

ASCII sketch:

```text
Vocv
  +
  |
  o---[ R0 ]---+---[ Rsei ]---+---[ Rct ]---+---[ Rw1 ]---+---[ Rw2 ]---o Vt
               |              |              |            |            |
               +----|| Csei---+              +----|| Cw1--+----|| Cw2--+
  |
 GND
```

In this formulation:

- `R0` captures the instantaneous ohmic contribution
- `Rsei || Csei` captures a fast SEI-related relaxation branch
- `Rct` is a charge-transfer-related resistive term
- `(Rw1 || Cw1)` and `(Rw2 || Cw2)` approximate the Warburg / diffusion tail with a small RC chain

### C. Model Training Workflow

Use `src/build_feature_table.py` to build a cycle-level ML table, optionally merge raw-derived time-domain features, then train models with:

- `src/train_swelling_models.py` for classical ML / XGBoost
- `src/train_swelling_deep.py` for MLP/CNN/LSTM
- `src/train_swelling_transformer.py` for Transformer models

## Quick Start

### 1) Frequency-domain baseline

```bash
python src/ecm_fit.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --recursive \
  --sheet "02_PreEIS" \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)-W1" \
  --guess "" \
  --out_dir "./data/ecm/ecm_w_cycle"
```

### 2) Build a labeled feature table

```bash
python src/build_feature_table.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_csv "./data/ml/feature_table.csv" \
  --min_cycle 5 \
  --max_cycle 200 \
  --future_k 20 \
  --soc_target 50 \
  --dcir_align_mode last_le
```

### 3) Device-side time-domain ECM features

```bash
python src/build_eis_time_domain_priors.py \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_csv "./data/ml/eis_time_domain_priors.csv"
```

```bash
python src/extract_device_ecm_features.py \
  --raw_dir "./dataset/raw_data" \
  --eis_prior_csv "./data/ml/eis_time_domain_priors.csv" \
  --prior_mode global \
  --fit_mode td_only \
  --cycle_mode last \
  --out_csv "./data/ml/device_ecm_with_priors.csv"
```

### 4) Train a classical model

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode current \
  --label_mode absolute \
  --max_input_cycle 50
```

## Project Structure

- `src/`: analysis, fitting, feature engineering, modeling, and visualization scripts
  - `cycle_plot.py`, `eis_plot.py`: workbook cycle and EIS plotting utilities
  - `ecm_fit.py`: equivalent-circuit-model fitting and fit-quality logging
  - `build_eis_time_domain_priors.py`: convert frequency-domain ECM outputs into time-domain priors (`init/lb/ub`)
  - `build_feature_table.py`: unified ML feature table builder from workbook data + ECM outputs
  - `extract_device_ecm_features.py`: extract time-domain ECM-inspired features from raw device `voltage/current/time`
  - `train_swelling_models.py`, `run_experiment_from_config.py`: classical ML/XGBoost training workflows
  - `train_swelling_deep.py`, `train_swelling_transformer.py`: optional neural-network and Transformer models
  - `plot_raw_maccor_signals.py`: visualize raw Maccor `capacity/energy/current/voltage` signals
  - `run_diff_soc_experiments.py`: automate multi-SOC correlation / importance experiments
  - `plot_*`, `check_*`, `filter_*`: diagnostics, correlation plots, alignment checks, feature importance, and result visualizations
- `configs/experiments/`: JSON configs for reproducible training runs
- `dataset/`: example location for workbook datasets and raw device datasets; large/private datasets can be stored elsewhere and passed by path
- `data/`: generated outputs such as ECM fits, feature tables, model results, plots, and logs
- `requirements.txt`: Python dependencies for the main analysis workflow
- `ecm_time_domain_fitting.md`, `time_domain_report.md`: design notes and progress report for the time-domain ECM workflow

## Environment

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install numpy pandas matplotlib openpyxl scipy impedance scikit-learn xgboost
```

## 1) cycle_plot.py

Generate cycle-related curves from sheets:
- `03-1_Cycle`
- `03-1_CycleMeasure`
- `03-1_CycleDCIR`

### CLI

```bash
python src/cycle_plot.py --xlsx <xlsx_path> --out <output_dir> [--serial <serial>]
```

### Parameters

- `--xlsx` (required): CL UDC xlsx path
- `--out` (required): output root directory
- `--serial` (optional): if provided, run only this serial; if omitted, auto-detect and run all serials

### Example

```bash
python src/cycle_plot.py \
  --xlsx "/path/to/CL-TC1-UDC.xlsx" \
  --out "./data/test_cycle"
```

### Output

Outputs are grouped by serial:

```text
data/test_cycle/<serial>/CL_DischargeCapacity_vs_Cycle__<serial>.png
data/test_cycle/<serial>/CL_Thickness2_vs_Cycle__<serial>.png
data/test_cycle/<serial>/CL_OCV_vs_Cycle__<serial>.png
data/test_cycle/<serial>/CL_ACIR_vs_Cycle__<serial>.png
data/test_cycle/<serial>/CL_DCIR_vs_Cycle_by_SOC__<serial>.png
data/test_cycle/<serial>/CL_OCV_vs_Cycle_by_SOC__<serial>.png
```

## 2) eis_plot.py

Scan workbook sheets and generate EIS plots by serial block.

### CLI

```bash
python src/eis_plot.py --xlsx <xlsx_path> --out <output_dir> [--serial <serial>] [--invert-imag]
```

### Parameters

- `--xlsx` (required): UDC xlsx path
- `--out` (required): output root directory
- `--serial` (optional): only process this serial
- `--invert-imag` (optional): Nyquist y-axis uses `-Imag`

### Example

```bash
python src/eis_plot.py \
  --xlsx "/path/to/test1.xlsx" \
  --out "./data/test_eis" \
  --invert-imag
```

### Output

```text
data/test_eis/<serial>/<sheet>__blkK_nyquist.png
data/test_eis/<serial>/<sheet>__blkK_bode_mag.png
data/test_eis/<serial>/<sheet>__blkK_bode_phase.png
```

## 3) ecm_fit.py

Fit EIS data to ECM model (default: no-Warburg 2-CPE):
- default circuit: `R0-p(R1,CPE1)-p(R2,CPE2)`

The script supports:
- single-file mode and directory-batch mode
- auto serial block traversal
- fallback block selection by valid numeric points
- multi-start fitting
- frequency filtering / high-frequency point dropping
- optional Warburg tail fitting (`W` / `Wo` / `Ws`)
- fit quality export (`json` + residual `csv`)

### CLI

```bash
python src/ecm_fit.py \
  [--xlsx <xlsx_path> | --xlsx_dir <xlsx_dir> [--recursive]] \
  [--sheet 02_PreEIS] \
  [--block 2] \
  [--serial <serial>] \
  [--circuit "R0-p(R1,CPE1)-p(R2,CPE2)"] \
  [--warburg none|W|Wo|Ws] \
  [--guess ""] \
  [--fmin <hz>] [--fmax <hz>] \
  [--drop_first_n <n>] \
  [--n_starts <n>] \
  [--weight_by_modulus] \
  --out_dir <output_dir>
```

### Common Parameters

- `--xlsx` / `--xlsx_dir`: provide exactly one input mode
- `--recursive`: recursively scan `--xlsx_dir` for `.xlsx`
- `--sheet`: target sheet (default `02_PreEIS`)
- `--block`: preferred block index; script can fallback to best block
- `--serial`: only run one serial, otherwise run all detected serials
- `--circuit`: ECM topology
- `--warburg`: append Warburg element to the circuit tail
  - `none`: no Warburg
  - `W`: semi-infinite Warburg
  - `Wo` / `Ws`: finite-length Warburg variants
- `--guess`: initial guess. Use empty string to trigger auto guess (`--guess ""`)
- `--auto_sign` / `--no_auto_sign`: imag sign policy
- `--fmin`, `--fmax`: frequency range filter
- `--drop_first_n`: remove top-N highest-frequency points
- `--n_starts`: multi-start count
- `--weight_by_modulus`: weighted fitting

### Recommended Example (no Warburg)

```bash
python src/ecm_fit.py \
  --xlsx "/path/to/test1.xlsx" \
  --sheet "02_PreEIS" \
  --block 2 \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)" \
  --guess "" \
  --fmin 0.1 \
  --drop_first_n 1 \
  --n_starts 8 \
  --weight_by_modulus \
  --out_dir "./data/test_ecm"
```

### Recommended Example (with Warburg tail)

```bash
python src/ecm_fit.py \
  --xlsx "/path/to/test1.xlsx" \
  --sheet "02_PreEIS" \
  --block 2 \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)" \
  --warburg W \
  --guess "" \
  --fmin 0.1 \
  --drop_first_n 1 \
  --n_starts 8 \
  --weight_by_modulus \
  --out_dir "./data/test_ecm_w"
```

### Directory Batch Example

```bash
python src/ecm_fit.py \
  --xlsx_dir "./dataset" \
  --recursive \
  --sheet "02_PreEIS" \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)" \
  --warburg W \
  --guess "" \
  --out_dir "./data/test_ecm_all"
```

### Output

```text
data/test_ecm/<group>/<xlsx_stem>/<serial>/nyquist_fit__<sheet>__block<k>.png
data/test_ecm/<group>/<xlsx_stem>/<serial>/fit_metrics__<sheet>__block<k>.json
data/test_ecm/<group>/<xlsx_stem>/<serial>/fit_residuals__<sheet>__block<k>.csv
data/test_ecm/<group>/<xlsx_stem>/<serial>/fit_result__<sheet>__block<k>.json
```

## How to Read ECM Fit Outputs

- `Params`: fitted ECM parameter values ordered by circuit string
- `fit_metrics*.json`:
  - `rmse_complex_ohm`: overall complex error (lower is better)
  - `nrmse_complex_percent_of_mean_absZ`: normalized error percentage
  - `r2_real`, `r2_imag`: goodness of fit for real/imag parts
- `fit_residuals*.csv`: pointwise residuals vs frequency

## Notes / Troubleshooting

- If no serial is detected, pass `--serial` explicitly.
- If EIS fitting looks unstable, first try:
  - `--guess ""`
  - `--drop_first_n 1`
  - `--fmin 0.1`
  - larger `--n_starts` (e.g., `8` or `10`)
- `eis_plot.py`/`ecm_fit.py` require EIS sheets with numeric frequency/real/imag data.
- If your environment cannot import dependencies, activate your project venv first.

## 4) ML Pipeline (ECM + Other Features)

This project now includes multiple scripts for swelling prediction modeling:

- `src/build_feature_table.py`: build a unified training table from:
  - ECM outputs (`fit_result`, `fit_metrics`)
  - cycle/capacity/thickness/DCIR/ACIR/OCV data from raw xlsx
- `src/train_swelling_models.py`: train/evaluate grouped models (`CL/FLC/HYCL`)
  with classic regressors including `Ridge`, `StepwiseLinear`, `RandomForest`,
  and `XGBoost(if installed)`.
- `src/run_experiment_from_config.py`: run `train_swelling_models.py` from a JSON config file.
- `src/train_swelling_deep.py`: train/evaluate grouped deep models (`MLP/CNN/LSTM`) with PyTorch.
- `src/train_swelling_transformer.py`: train/evaluate grouped Transformer model with PyTorch.
- `src/benchmark_models.py`: batch benchmark runner for `train_swelling_models.py`
  across multiple `model_set x feature_set` combinations.
- `src/plot_feature_corr.py`: plot feature correlation matrix heatmap from `feature_table.csv`.
- `src/plot_predictions_scatter.py`: plot `y_true` vs `y_pred` scatter plots from `predictions__*.csv`.
- `src/plot_stepwise_regression.py`: visualize `stepwise_trace__*.csv` as stepwise path, improvement bars,
  and feature-entry heatmap.
- `src/plot_permutation_importance.py`: plot permutation importance from a trained classic model setup.
- `src/plot_incremental_cv_mae.py`: plot incremental CV-MAE curves under a specified feature order.
- `src/plot_ecm_param_distributions.py`: summarize and visualize ECM parameter ranges/distributions from `fit_result__*.json`.
- `src/check_ecm_dcir_alignment.py`: check exact cycle overlap between ECM measurement cycles and DCIR cycles.
- `src/plot_ecm_dcir_cycle_coverage.py`: visualize ECM/DCIR cycle coverage per cell and in aggregate.
- `src/parse_raw_maccor.py`: parse raw Maccor text exports (`dataset/raw_data`) and extract
  row/cycle summaries including `EVTemp (C)` / `EVHum (%)`, with optional merge into `feature_table.csv`.
- `src/build_eis_time_domain_priors.py`: build an EIS-derived prior table for the time-domain ECM fitter
  from `fit_result__*.json`.
- `src/extract_device_ecm_features.py`: extract device-side time-domain ECM-inspired features from raw Maccor
  `current/voltage/time`, with optional EIS-prior alignment and optional merge into a feature table.
- `src/plot_raw_maccor_signals.py`: visualize raw Maccor `capacity`, `energy`, `current`, and `voltage` signals.
- `src/run_diff_soc_experiments.py`: automate repeated build/train/correlation/permutation runs across DCIR SOC targets.
- `src/filter_feature_table_outliers.py`: optional plug-in for outlier detection/removal on feature tables.
  Default mode is report-only (no row deletion).

### Extra Dependencies

```bash
pip install scikit-learn xgboost
```

For deep models:

```bash
pip install torch
```

### Step A0b (Optional): Build EIS-Derived Priors for Time-Domain ECM Fitting

This converts `ecm_fit.py` outputs into a serial/cycle-aligned prior table that contains:

- initial guesses,
- lower bounds,
- upper bounds,

for the time-domain ECM fitter.

```bash
python src/build_eis_time_domain_priors.py \
  --ecm_dir "./data/ecm_w_cycle" \
  --out_csv "./data/ml/eis_time_domain_priors.csv"
```

Useful option:

- `--buffer_frac`: relative margin used to expand EIS-derived values into default bounds

Typical output columns include:

- `prior_Rs_ohm`
- `prior_Rsei_ohm`
- `prior_Rct_ohm`
- `prior_Rw1_ohm`
- `prior_Rw2_ohm`
- `prior_tau_Rsei_s`
- `prior_tau_Rw1_s`
- `prior_tau_Rw2_s`
- and corresponding `*_init`, `*_lb`, `*_ub`

### Step A: Build Unified Feature Table

```bash
python src/build_feature_table.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_csv "./data/ml/feature_table.csv" \
  --min_cycle 5 \
  --max_cycle 200 \
  --future_k 20 \
  --soc_target 50 \
  --dcir_align_mode last_le
```

Output: `./data/ml/feature_table.csv`

Useful options:
- `--soc_target`: target SOC used to choose the DCIR slice.
- `--dcir_align_mode last_le|exact`:
  - `last_le` (default): use the latest `cycle_target <= cycle_t`
  - `exact`: require `cycle_target == cycle_t`
- `--log_file`: save a copy of stdout/stderr while keeping terminal output.

### Step A0 (Optional): Parse `dataset/raw_data` and add temperature features

If you want to use raw Maccor temperature as ML features:

```bash
python src/parse_raw_maccor.py \
  --raw_dir "./dataset/raw_data" \
  --out_row_csv "./data/ml/raw_maccor_rows.csv" \
  --out_cycle_csv "./data/ml/raw_maccor_cycle_summary.csv" \
  --feature_table_csv "./data/ml/feature_table.csv" \
  --out_feature_table_csv "./data/ml/feature_table_with_raw_temp.csv"
```

Then use `feature_table_with_raw_temp.csv` as input to `train_swelling_models.py`.

### Step A0c (Optional): Extract Device-Side Time-Domain ECM Features from Raw Data

This step uses raw Maccor-like `current/voltage/time` data to fit time-domain ECM-inspired features.

The simplified time-domain ECM used in this branch is:

```text
Vocv - R0 - (Rsei || Csei) - Rct - (Rw1 || Cw1) - (Rw2 || Cw2)
```

ASCII sketch:

```text
Vocv
  +
  |
  o---[ R0 ]---+---[ Rsei ]---+---[ Rct ]---+---[ Rw1 ]---+---[ Rw2 ]---o Vt
               |              |              |            |            |
               +----|| Csei---+              +----|| Cw1--+----|| Cw2--+
  |
 GND
```

Here:

- `R0` captures the instantaneous ohmic contribution,
- `Rsei || Csei` captures a fast SEI-related relaxation branch,
- `Rct` is a charge-transfer-related resistive term,
- `(Rw1 || Cw1)` and `(Rw2 || Cw2)` are a small RC-chain approximation of the diffusion / Warburg-like tail.

```bash
python src/extract_device_ecm_features.py \
  --raw_dir "./dataset/raw_data" \
  --eis_prior_csv "./data/ml/eis_time_domain_priors.csv" \
  --prior_align_mode last_le \
  --fit_mode td_only \
  --cycle_mode last \
  --out_csv "./data/ml/device_ecm_with_priors.csv"
```

Useful options:

- `--eis_prior_csv`: optional prior table from `build_eis_time_domain_priors.py`
- `--prior_align_mode last_le|exact`: how to align priors to raw-data `cycle_c`
- `--fit_mode full|td_only`
  - `full`: exploratory fits + constrained fitter
  - `td_only`: mentor-style constrained time-domain fitter only
- `--cycle_mode all|first|last`: whether to keep all raw cycles or one representative cycle per file

Representative outputs include:

- `feat_dev_r0_proxy_ohm`
- `feat_dev_td_Rsei_ohm`
- `feat_dev_td_Rw1_ohm`
- `feat_dev_td_Rw2_ohm`
- `feat_dev_td_R_diff_total_ohm`
- `feat_dev_td_R_total_proxy_ohm`
- `feat_dev_td_prior_used`

To merge these features directly into a feature table:

```bash
python src/extract_device_ecm_features.py \
  --raw_dir "./dataset/raw_data" \
  --eis_prior_csv "./data/ml/eis_time_domain_priors.csv" \
  --out_csv "./data/ml/device_ecm_with_priors.csv" \
  --feature_table_csv "./data/ml/feature_table.csv" \
  --out_feature_table_csv "./data/ml/feature_table_with_device_ecm.csv" \
  --align_mode last_le
```

### Step A1 (Optional): Outlier Detection / Removal (Plug-in)

This step is optional and can be enabled or skipped as needed.
By default, the script only reports outliers and does not modify your table.

```bash
python src/filter_feature_table_outliers.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/outlier_report" \
  --sample_mode future_delta_TK \
  --max_input_cycle 50 \
  --group_tag HYCL
```

To actually drop flagged outliers and export a cleaned table:

```bash
python src/filter_feature_table_outliers.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/outlier_report_drop" \
  --sample_mode future_delta_TK \
  --max_input_cycle 50 \
  --group_tag HYCL \
  --apply_drop \
  --out_clean_csv "./data/ml/feature_table_cleaned.csv"
```

Key options:
- `--method robust|iqr|combined`: detector type.
- `--combined_rule two_of_three|any`: how to combine detectors in `combined` mode.
  - `two_of_three` (default): balanced, less aggressive
  - `any`: aggressive
- `--robust_z_thresh`, `--iqr_k`, `--iqr_min_count`, `--mahal_q`: sensitivity controls.

### Step B: Train & Evaluate (Grouped by CL/FLC/HYCL)

`target_mode` and `label_mode` are parameterized so you can compare:
- absolute thickness vs delta thickness
- current-cycle estimation, fixed cycle T prediction, and future T->T+K prediction

#### 1) Current-cycle absolute thickness

This estimates thickness at the same cycle as the input features.

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode current \
  --label_mode absolute \
  --max_input_cycle 50
```

#### 2) Fixed cycle T, absolute thickness

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50
```

You can expand models and feature subsets with:

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results_ext" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50 \
  --model_set extended \
  --feature_set variance \
  --variance_top_n 16 \
  --run_tag "extended_variance"
```

To inspect feature-by-feature entry order with stepwise regression:

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results_stepwise" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50 \
  --models StepwiseLinear \
  --feature_set variance \
  --variance_top_n 16 \
  --stepwise_max_features 8 \
  --stepwise_min_improvement 0.0001 \
  --run_tag "stepwise_v1"
```

#### 3) Fixed cycle T, delta thickness

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode fixed_T \
  --label_mode delta \
  --T 100 \
  --max_input_cycle 50
```

#### 4) Future T->T+K, absolute thickness at t+K

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode future_delta_TK \
  --label_mode absolute \
  --future_k 20 \
  --max_input_cycle 50
```

#### 5) Future T->T+K, delta thickness (t+K minus t)

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode future_delta_TK \
  --label_mode delta \
  --future_k 20 \
  --max_input_cycle 50
```

### ML Outputs

```text
data/ml/results/results__<target_mode>__<label_mode>__<mode_tag>.csv
data/ml/results/predictions__<target_mode>__<label_mode>__<mode_tag>.csv
data/ml/results/stepwise_trace__<target_mode>__<label_mode>__<mode_tag>.csv
data/ml/results/run_meta__<target_mode>__<label_mode>__<mode_tag>.json
```

Each result CSV includes RMSE and MAE per model per group (`CL/FLC/HYCL`).

`train_swelling_models.py` supports:
- `--model_set basic|extended|all`
  - `basic`: Ridge + RandomForest + XGBoost(if available)
  - `extended`: basic + Dummy + Linear + StepwiseLinear + PCR + PLSR + GaussianProcess + MLP
- `--feature_set full|variance|discharge|ecm|custom`
- `--variance_top_n` for `variance`
- `--custom_features` for `custom`
- `--sample_mode anchor|rowwise`
  - `anchor`: one sample per cell
  - `rowwise`: for `fixed_T`, one sample per row up to `max_input_cycle`, with the same cell's fixed-`T` thickness as the target
- `--target_transform none|log` for optional log-transform on positive absolute targets
- `--stepwise_max_features`, `--stepwise_min_improvement`, `--stepwise_cv_splits` for `StepwiseLinear`
- `--xgb_n_estimators`, `--xgb_max_depth`, `--xgb_learning_rate`,
  `--xgb_subsample`, `--xgb_colsample_bytree`, `--xgb_min_child_weight`,
  `--xgb_reg_alpha`, `--xgb_reg_lambda` for XGBoost tuning
- `--run_tag` to append a suffix in output file names
- `--log_file` to tee stdout/stderr into a file

### Step B0: Config-Driven Experiments

You can run classic-model experiments from a JSON config:

```bash
python src/run_experiment_from_config.py \
  --config configs/experiments/hycl_xgb_t03_slow.json \
  --dry_run
```

Then execute it directly:

```bash
python src/run_experiment_from_config.py \
  --config configs/experiments/hycl_xgb_t03_slow.json
```

### Step B1: Batch Benchmark (Optional)

Run multiple model/feature combinations in one command:

```bash
python src/benchmark_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/benchmark" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50 \
  --model_sets "basic,extended" \
  --feature_sets "full,variance,discharge"
```

Batch outputs:
- `benchmark_runs.csv`: run ledger + status
- `benchmark_results_aggregate.csv`: concatenated `results__*.csv` from successful runs
- per-run logs under each benchmark subfolder

### Step B2: Deep Models (Phase 2: MLP/CNN/LSTM)

Train deep models with the same target modes and output format:

```bash
python src/train_swelling_deep.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results_deep" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50 \
  --groups HYCL \
  --models mlp,cnn,lstm \
  --feature_set variance \
  --variance_top_n 20 \
  --epochs 120 \
  --batch_size 32 \
  --lr 1e-3 \
  --hidden_dim 64 \
  --run_tag "deep_v1"
```

Useful options:
- `--groups`: choose subset groups, e.g. `HYCL` or `CL,FLC,HYCL`
- `--feature_set`: `full|variance|discharge|ecm|custom`
- `--custom_features`: comma list when `--feature_set custom`
- `--models`: comma list from `mlp,cnn,lstm`

### Step B3: Transformer Model (Phase 2: Transformer)

Train Transformer with the same grouped split/output format:

```bash
python src/train_swelling_transformer.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results_transformer" \
  --target_mode fixed_T \
  --label_mode absolute \
  --T 100 \
  --max_input_cycle 50 \
  --groups HYCL \
  --feature_set variance \
  --variance_top_n 20 \
  --epochs 160 \
  --batch_size 32 \
  --lr 5e-4 \
  --hidden_dim 64 \
  --n_heads 4 \
  --n_layers 2 \
  --ff_dim 128 \
  --run_tag "transformer_v1"
```

Useful options:
- `--groups`: choose subset groups, e.g. `HYCL` or `CL,FLC,HYCL`
- `--feature_set`: `full|variance|discharge|ecm|custom`
- `--custom_features`: comma list when `--feature_set custom`
- `--hidden_dim`: Transformer `d_model` (must be divisible by `--n_heads`)
- `--n_heads`, `--n_layers`, `--ff_dim`: Transformer architecture settings

### Step B4: Visualize Raw Maccor Signals (Optional)

To inspect the raw device-data source before time-domain fitting:

```bash
python src/plot_raw_maccor_signals.py \
  --raw_dir "./dataset/raw_data" \
  --out_dir "./data/raw_viz"
```

Typical outputs:

```text
data/raw_viz/plots/per_file/*.png
data/raw_viz/plots/overlay_all_files.png
data/raw_viz/plots/overlay_by_signal/*.png
data/raw_viz/file_signal_summary.csv
```

### Step B5: Diff-SOC Analysis (Optional)

To compare DCIR/ECM behavior across multiple SOC targets:

```bash
python src/run_diff_soc_experiments.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --ecm_dir "./data/ecm_w_cycle" \
  --out_root "./data/diff_soc_runs" \
  --socs "0,10,20,30,40,50,60,70,80,90,100" \
  --n_repeats 30
```

This script automates:

- feature-table building,
- ECM-complete filtering,
- model training,
- permutation importance plotting,
- correlation matrix plotting,
- SOC-level summary export.

### How to Read ML Result Files

#### `results__*.csv`

Each row is one model result under one group (`CL`/`FLC`/`HYCL`), with key fields:

- `model`: model name (`Ridge`, `RandomForest`, `XGBoost` if installed)
- `group_tag`: dataset group
- `rmse`: root mean squared error (lower is better)
- `mae`: mean absolute error (lower is better)
- `n_train`, `n_test`: sample counts in train/test split
- `n_cells_train`, `n_cells_test`: unique cell counts in train/test
- `n_features_used`: numeric feature count actually used in that group
- `selected_features`: final selected feature list for models that do feature selection
- `target_mode`, `label_mode`, `mode_tag`, `max_input_cycle`: run context

#### `run_meta__*.json`

This is the run configuration and feature snapshot for reproducibility:

- `table_csv`: input feature table path
- `target_mode`: `current`, `fixed_T`, or `future_delta_TK`
- `label_mode`: `absolute` or `delta`
- `target_transform`: `none` or `log`
- `T`: target cycle for `fixed_T`
- `future_k`: K value for `future_delta_TK`
- `max_input_cycle`: max cycle allowed for input features
- `seed`: random seed
- `test_size`: test split ratio (grouped by `cell_key`)
- `feature_count`: number of features used
- `feature_columns`: full feature column list used in training
- `stepwise_*`: stepwise search configuration when `StepwiseLinear` is enabled

#### `predictions__*.csv`

Each row is one test sample prediction, useful for direct comparison between predicted and true thickness:

- `model`: model name
- `cell_key`: cell/sample identifier used for grouped split
- `serial`: serial number if available
- `group_tag`: `CL`, `FLC`, or `HYCL`
- `cycle_t`: input anchor cycle used by the model
- `target_cycle`: cycle of the target thickness being predicted
- `label_col`: target column used internally (`target_abs` or `target_delta`)
- `y_true`: true target value
- `y_pred`: predicted target value
- `abs_error`: absolute prediction error
- `target_mode`, `label_mode`, `mode_tag`, `max_input_cycle`: run context

#### `stepwise_trace__*.csv`

Each row is one accepted step from `StepwiseLinear`, useful for understanding
what the model discovered incrementally:

- `group_tag`: dataset group
- `step`: selection order
- `feature_name`: feature added at this step
- `cv_mae`: train-only cross-validated MAE after adding this feature
- `improvement`: CV-MAE gain versus previous step

### Step B6: Visualize Stepwise Regression

If you enabled `StepwiseLinear`, you can visualize the feature-entry process:

```bash
python src/plot_stepwise_regression.py \
  --trace_csv "./data/ml/results_stepwise/stepwise_trace__fixed_T__absolute__fixedT_100__stepwise_v1.csv" \
  --out_png "./data/ml/results_stepwise/stepwise_viz.png" \
  --mode all
```

Outputs with `--mode all`:
- `stepwise_viz__path.png`: CV-MAE vs step, annotated with feature names
- `stepwise_viz__improvement.png`: per-step improvement bar chart
- `stepwise_viz__heatmap.png`: feature entry-order heatmap across groups/models

### Step C: Plot Feature Correlation Matrices

By default, the script saves two heatmaps:
- feature-only correlation matrix
- feature + target correlation matrix

If `--out_png` is `./data/ml/feature_corr.png`, the outputs will be:

```text
./data/ml/feature_corr__features.png
./data/ml/feature_corr__features_targets.png
```

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_png "./data/ml/feature_corr.png" \
  --method pearson \
  --max_features 40 \
  --annot
```

If you only want the feature-only matrix:

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_png "./data/ml/feature_corr.png" \
  --mode features
```

If you only want the feature + target matrix:

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_png "./data/ml/feature_corr.png" \
  --mode features_targets
```

### Step D: Plot Prediction Scatter (`y_true` vs `y_pred`)

By default, the script saves three scatter plots:
- combined across all rows
- split by model
- split by group

If `--out_png` is `./data/ml/pred_scatter.png`, the outputs will be:

```text
./data/ml/pred_scatter__combined.png
./data/ml/pred_scatter__by_model.png
./data/ml/pred_scatter__by_group.png
```

```bash
python src/plot_predictions_scatter.py \
  --pred_csv "./data/ml/results/predictions__fixed_T__absolute__fixedT_100.csv" \
  --out_png "./data/ml/pred_scatter.png"
```

If you only want one view:

```bash
python src/plot_predictions_scatter.py \
  --pred_csv "./data/ml/results/predictions__fixed_T__absolute__fixedT_100.csv" \
  --out_png "./data/ml/pred_scatter.png" \
  --mode by_model
```

## End-to-End Example

This section gives one reproducible command chain from ECM fitting to model
training and related visualizations. Paths are repo-relative examples:

- source workbooks: `./dataset/udc_xlsx`
- cached ECM fits: `./data/ecm/ecm_w_cycle`
- ML tables/results: `./data/ml`
- run logs: `./data/logs`

### 1) ECM fitting

```bash
python src/ecm_fit.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --recursive \
  --sheet auto \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)-W1" \
  --guess "" \
  --merge_serial_plots \
  --skip_existing \
  --out_dir "./data/ecm/ecm_w_cycle" \
  --log_file "./data/logs/ecm_fit_cycle.log"
```

### 1b) Check ECM fitting progress

`ecm_fit.py` appends one JSON record per completed, skipped, or failed fitting
task to `ecm_progress.jsonl`.

```bash
python - <<'PY'
import json
from collections import Counter

p = "data/ecm/ecm_w_cycle/ecm_progress.jsonl"
cnt = Counter()
with open(p, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            cnt[json.loads(line)["status"]] += 1
print(cnt)
PY
```

### 2) Build feature table

```bash
python src/build_feature_table.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_csv "./data/ml/feature_table.csv" \
  --min_cycle 5 \
  --max_cycle 200 \
  --future_k 20 \
  --soc_target 50 \
  --dcir_align_mode last_le \
  --log_file "./data/logs/build_feature_table.log"
```

`build_feature_table.py` automatically applies the ECM bad-fit filter before
writing ECM-derived features.

### 2b) Create ECM-complete feature table

This optional subset keeps only rows with the core ECM-derived features present.
Because Step 2 leaves failed or unusable ECM fits as missing values, this table
is useful when you want to train models that compare samples on the same ECM
feature set.

```bash
python - <<'PY'
import pandas as pd

src = "data/ml/feature_table.csv"
dst = "data/ml/feature_table_ecm_complete.csv"
df = pd.read_csv(src)
ecm_cols = [
    "feat_Rs_ohm",
    "feat_nsei",
    "feat_ndl",
    "feat_R_total_ohm",
    "feat_sigma",
]
df.dropna(subset=ecm_cols).to_csv(dst, index=False)
print("saved:", dst)
PY
```

### 2c) Optional: Add device-side time-domain ECM features

This branch keeps the existing lab-side feature table, but augments it with
time-domain ECM-inspired features extracted from raw device data.

1. Build EIS-derived priors:

```bash
python src/build_eis_time_domain_priors.py \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_csv "./data/ml/eis_time_domain_priors.csv"
```

2. Extract device-side time-domain features and merge them directly into the feature table:

```bash
python src/extract_device_ecm_features.py \
  --raw_dir "./dataset/raw_data" \
  --eis_prior_csv "./data/ml/eis_time_domain_priors.csv" \
  --prior_align_mode last_le \
  --fit_mode td_only \
  --cycle_mode last \
  --out_csv "./data/ml/device_ecm_with_priors.csv" \
  --feature_table_csv "./data/ml/feature_table_ecm_complete.csv" \
  --out_feature_table_csv "./data/ml/feature_table_ecm_complete_with_device.csv" \
  --align_mode last_le
```

Then use `feature_table_ecm_complete_with_device.csv` in downstream training.

### 3) Train a tuned XGBoost experiment from config

```bash
python src/run_experiment_from_config.py \
  --config configs/experiments/config.json
```

Or run the same experiment directly from the command line:

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table_ecm_complete.csv" \
  --out_dir "./data/ml/experiments/xgb_t11_lighter_reg" \
  --target_mode current \
  --sample_mode rowwise \
  --label_mode absolute \
  --target_transform log \
  --max_input_cycle 50 \
  --model_set basic \
  --models "XGBoost" \
  --feature_set custom \
  --custom_features "feat_cycle_t,feat_Rs_ohm,feat_nsei,feat_ndl,feat_R_total_ohm,feat_sigma,feat_capacity_t,feat_capacity_slope_10,feat_dcir_soc_t" \
  --xgb_n_estimators 1200 \
  --xgb_max_depth 4 \
  --xgb_learning_rate 0.015 \
  --xgb_subsample 0.85 \
  --xgb_colsample_bytree 0.85 \
  --xgb_min_child_weight 2 \
  --xgb_reg_alpha 0.05 \
  --xgb_reg_lambda 2.0 \
  --run_tag "xgb_t11_lighter_reg" \
  --log_file "./data/logs/xgb_t11_lighter_reg.log"
```

If you want to include the device-side time-domain ECM branch as additional features:

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table_ecm_complete_with_device.csv" \
  --out_dir "./data/ml/experiments/xgb_td_device" \
  --target_mode current \
  --sample_mode rowwise \
  --label_mode absolute \
  --target_transform log \
  --max_input_cycle 120 \
  --model_set basic \
  --models "XGBoost" \
  --feature_set custom \
  --custom_features "feat_cycle_t,feat_Rs_ohm,feat_nsei,feat_ndl,feat_R_total_ohm,feat_sigma,feat_capacity_t,feat_capacity_slope_10,feat_dcir_soc_t,feat_dev_td_R_total_proxy_ohm,feat_dev_td_prior_used" \
  --run_tag "xgb_td_device" \
  --log_file "./data/logs/xgb_td_device.log"
```

### 4) Permutation importance

```bash
python src/plot_permutation_importance.py \
  --table_csv "./data/ml/feature_table_ecm_complete.csv" \
  --out_dir "./data/ml/experiments/xgb_t11_lighter_reg" \
  --target_mode current \
  --sample_mode rowwise \
  --label_mode absolute \
  --target_transform log \
  --group_tag HYCL \
  --model XGBoost \
  --custom_features "feat_cycle_t,feat_Rs_ohm,feat_nsei,feat_ndl,feat_R_total_ohm,feat_sigma,feat_capacity_t,feat_capacity_slope_10,feat_dcir_soc_t" \
  --max_input_cycle 50 \
  --xgb_n_estimators 1200 \
  --xgb_max_depth 4 \
  --xgb_learning_rate 0.015 \
  --xgb_subsample 0.85 \
  --xgb_colsample_bytree 0.85 \
  --xgb_min_child_weight 2 \
  --xgb_reg_alpha 0.05 \
  --xgb_reg_lambda 2.0 \
  --n_repeats 30 \
  --metric mae
```

### 5) Incremental CV-MAE visualization

```bash
python src/plot_incremental_cv_mae.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/incremental_cv" \
  --target_mode fixed_T \
  --label_mode absolute \
  --target_transform log \
  --group_tag HYCL \
  --model Ridge \
  --custom_features "feat_cycle_t,feat_capacity_t,feat_capacity_slope_10,feat_dcir_soc_t,feat_Rs_ohm,feat_nsei,feat_ndl,feat_R_total_ohm,feat_sigma" \
  --T 100 \
  --max_input_cycle 50 \
  --cv_splits 5
```

### 6) ECM parameter distributions

```bash
python src/plot_ecm_param_distributions.py \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_dir "./data/ecm/param_distributions" \
  --group_tag HYCL \
  --sheet 03-4_EIS \
  --rmse_max 1.0 \
  --title "ECM Parameter Distributions"
```

### 7) ECM/DCIR exact-alignment check

```bash
python src/check_ecm_dcir_alignment.py \
  --xlsx_dir "./dataset/udc_xlsx" \
  --ecm_dir "./data/ecm/ecm_w_cycle" \
  --out_dir "./data/ecm/alignment_check" \
  --group_tag HYCL \
  --soc_target 50 \
  --sheet 03-4_EIS \
  --rmse_max 1.0
```

### 8) ECM/DCIR cycle coverage visualization

```bash
python src/plot_ecm_dcir_cycle_coverage.py \
  --overview_csv "./data/ecm/alignment_check/ecm_dcir_exact_alignment__HYCL__overview.csv" \
  --out_dir "./data/ecm/alignment_check/plots" \
  --title_prefix "HYCL ECM vs DCIR Cycle Coverage"
```

### 9) Feature correlation matrix visualization

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table_ecm_complete.csv" \
  --out_png "./data/ml/corr_selected_ecm_cap_dcir.png" \
  --columns "feat_Rs_ohm,feat_dcir_soc_t,feat_R_total_ohm,feat_capacity_t,feat_capacity_slope_10,feat_cycle_t,feat_nsei,feat_ndl,feat_sigma,y_abs_thickness_t" \
  --group_tag HYCL \
  --max_cycle 50 \
  --method spearman \
  --annot \
  --mode features_targets
```
