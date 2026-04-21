# battery-swelling-prediction

Utilities for battery UDC Excel analysis:
- Cycle/aging curves (`src/cycle_plot.py`)
- EIS plotting (`src/eis_plot.py`)
- ECM fitting (`src/ecm_fit.py`)

## Project Structure

- `src/cycle_plot.py`: capacity/thickness/OCV/ACIR/DCIR plotting by serial
- `src/eis_plot.py`: Nyquist + Bode plotting from EIS sheets
- `src/ecm_fit.py`: equivalent circuit fitting and fit-quality outputs
- `dataset/`: original UDC files
- `data/`: generated plots/results

## Environment

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install numpy pandas matplotlib openpyxl impedance
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

### Step A: Build Unified Feature Table

```bash
python src/build_feature_table.py \
  --xlsx_dir "./dataset/OneDrive_1_2-20-2026" \
  --ecm_dir "./data/test_ecm_all4" \
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

### Step A1 (Optional): Outlier Detection / Removal (Plug-in)

This step is optional and can be enabled or skipped as needed.
By default, the script only reports outliers and does not modify your table.

```bash
python src/filter_feature_table_outliers.py \
  --table_csv "./data/ml/test15/feature_table_test15.csv" \
  --out_dir "./data/ml/test15/outlier_report" \
  --sample_mode future_delta_TK \
  --max_input_cycle 50 \
  --group_tag HYCL
```

To actually drop flagged outliers and export a cleaned table:

```bash
python src/filter_feature_table_outliers.py \
  --table_csv "./data/ml/test15/feature_table_test15.csv" \
  --out_dir "./data/ml/test15/outlier_report_drop" \
  --sample_mode future_delta_TK \
  --max_input_cycle 50 \
  --group_tag HYCL \
  --apply_drop \
  --out_clean_csv "./data/ml/test15/feature_table_test15_cleaned.csv"
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
- fixed cycle T vs future T->T+K

#### 1) Fixed cycle T, absolute thickness

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

#### 2) Fixed cycle T, delta thickness

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode fixed_T \
  --label_mode delta \
  --T 100 \
  --max_input_cycle 50
```

#### 3) Future T->T+K, absolute thickness at t+K

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_dir "./data/ml/results" \
  --target_mode future_delta_TK \
  --label_mode absolute \
  --future_k 20 \
  --max_input_cycle 50
```

#### 4) Future T->T+K, delta thickness (t+K minus t)

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
  - `extended`: basic + Dummy + Linear + StepwiseLinear + PCR + PLSR + GaussianProcess
- `--feature_set full|variance|discharge|ecm|custom`
- `--variance_top_n` for `variance`
- `--custom_features` for `custom`
- `--sample_mode anchor|rowwise`
  - `anchor`: one sample per cell
  - `rowwise`: one sample per row up to `max_input_cycle`
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
  --table_csv "./data/ml/hycl_od/feature_table_hycl_pruned.csv" \
  --out_dir "./data/ml/hycl_od/results_deep" \
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
  --table_csv "./data/ml/hycl_od/feature_table_hycl_pruned.csv" \
  --out_dir "./data/ml/hycl_od/results_transformer" \
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
- `target_mode`: `fixed_T` or `future_delta_TK`
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

### Step B4: Visualize Stepwise Regression

If you enabled `StepwiseLinear`, you can visualize the feature-entry process:

```bash
python src/plot_stepwise_regression.py \
  --trace_csv "./data/ml3/compare_hycl/onedrive_allmodels/results_classic_stepwise/stepwise_trace__fixed_T__absolute__fixedT_100__stepwise_v1.csv" \
  --out_png "./data/ml3/compare_hycl/onedrive_allmodels/results_classic_stepwise/stepwise_viz.png" \
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
training and related visualizations. Paths below match the current repo layout.

### 1) ECM fitting

```bash
python src/ecm_fit.py \
  --xlsx_dir "./dataset/OneDrive_1_2-20-2026" \
  --recursive \
  --sheet auto \
  --circuit "R0-p(R1,CPE1)-p(R2,CPE2)" \
  --warburg W \
  --guess "" \
  --merge_serial_plots \
  --skip_existing \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/ecm_w_cycle" \
  --log_file "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/logs/ecm_fit_cycle.log"
```

### 2) Build feature table

```bash
python src/build_feature_table.py \
  --xlsx_dir "./dataset/OneDrive_1_2-20-2026" \
  --ecm_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/ecm_w_cycle" \
  --out_csv "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/feature_table_h_cycle_ecm.csv" \
  --min_cycle 5 \
  --max_cycle 200 \
  --future_k 20 \
  --soc_target 50 \
  --dcir_align_mode last_le \
  --log_file "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/logs/build_feature_table_cycle.log"
```

### 3) Train a tuned XGBoost experiment from config

```bash
python src/run_experiment_from_config.py \
  --config configs/experiments/hycl_xgb_t11_lighter_reg.json
```

Or run the same experiment directly from the command line:

```bash
python src/train_swelling_models.py \
  --table_csv "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/feature_table_h_cycle_ecm_complete6.csv" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/xgb_tuning/t11_lighter_reg" \
  --target_mode fixed_T \
  --sample_mode rowwise \
  --label_mode absolute \
  --target_transform log \
  --T 100 \
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
  --log_file "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/logs/xgb_t11_lighter_reg.log"
```

### 4) Permutation importance

```bash
python src/plot_permutation_importance.py \
  --table_csv "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/feature_table_h_cycle_ecm.csv" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/perm_importance_ecm6_plus_cap_dcir" \
  --target_mode fixed_T \
  --label_mode absolute \
  --target_transform log \
  --group_tag HYCL \
  --model XGBoost \
  --custom_features "feat_cycle_t,feat_Rs_ohm,feat_nsei,feat_ndl,feat_R_total_ohm,feat_sigma,feat_capacity_t,feat_capacity_slope_10,feat_dcir_soc_t" \
  --T 100 \
  --max_input_cycle 50 \
  --n_repeats 30 \
  --metric mae
```

### 5) Incremental CV-MAE visualization

```bash
python src/plot_incremental_cv_mae.py \
  --table_csv "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/feature_table_h_cycle_ecm.csv" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/incremental_cv_ecm6_plus_cap_dcir" \
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
  --ecm_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/ecm_w_cycle" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/ecm_param_distributions" \
  --group_tag HYCL \
  --sheet 03-4_EIS \
  --rmse_max 1.0 \
  --title "ECM Parameter Distributions"
```

### 7) ECM/DCIR exact-alignment check

```bash
python src/check_ecm_dcir_alignment.py \
  --xlsx_dir "./dataset/OneDrive_1_2-20-2026" \
  --ecm_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/ecm_w_cycle" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/alignment_check" \
  --group_tag HYCL \
  --soc_target 50 \
  --sheet 03-4_EIS \
  --rmse_max 1.0
```

### 8) ECM/DCIR cycle coverage visualization

```bash
python src/plot_ecm_dcir_cycle_coverage.py \
  --overview_csv "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/alignment_check/ecm_dcir_exact_alignment__HYCL__overview.csv" \
  --out_dir "./data/ml3/compare_hycl/onedrive_h_cycle_ecm/alignment_check/plots" \
  --title_prefix "HYCL ECM vs DCIR Cycle Coverage"
```

### 9) Correlation matrix for selected features

```bash
python - <<'PY'
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

p = "data/ml3/compare_hycl/onedrive_h_cycle_ecm/feature_table_h_cycle_ecm.csv"
df = pd.read_csv(p)
sub = df[(df["group_tag"] == "HYCL") & (df["cycle_t"] <= 50)].copy()

cols = [
    "feat_Rs_ohm",
    "feat_dcir_soc_t",
    "feat_R_total_ohm",
    "feat_capacity_t",
    "feat_capacity_slope_10",
    "feat_cycle_t",
    "feat_nsei",
    "feat_ndl",
    "feat_sigma",
    "y_abs_thickness_t",
]

corr = sub[cols].corr(numeric_only=True)

out_csv = "data/ml3/compare_hycl/onedrive_h_cycle_ecm/corr_matrix_ecm6_plus_cap_dcir.csv"
out_png = "data/ml3/compare_hycl/onedrive_h_cycle_ecm/corr_matrix_ecm6_plus_cap_dcir.png"

corr.to_csv(out_csv)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix: HYCL ECM + capacity + DCIR")
plt.tight_layout()
plt.savefig(out_png, dpi=200)
print("saved:", out_csv)
print("saved:", out_png)
PY
```
