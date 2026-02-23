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

This project now includes two scripts for swelling prediction modeling:

- `src/build_feature_table.py`: build a unified training table from:
  - ECM outputs (`fit_result`, `fit_metrics`)
  - cycle/capacity/thickness/DCIR/ACIR/OCV data from raw xlsx
- `src/train_swelling_models.py`: train/evaluate grouped models (`CL/FLC/HYCL`)
  with `Ridge + RandomForest + XGBoost(if installed)`.
- `src/plot_feature_corr.py`: plot feature correlation matrix heatmap from `feature_table.csv`.

### Extra Dependencies

```bash
pip install scikit-learn xgboost
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
  --soc_target 50
```

Output: `./data/ml/feature_table.csv`

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
data/ml/results/run_meta__<target_mode>__<label_mode>__<mode_tag>.json
```

Each result CSV includes RMSE and MAE per model per group (`CL/FLC/HYCL`).

### Step C: Plot Feature Correlation Matrix

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_png "./data/ml/feature_corr.png" \
  --method pearson \
  --max_features 40 \
  --annot
```

If you also want target columns in the matrix:

```bash
python src/plot_feature_corr.py \
  --table_csv "./data/ml/feature_table.csv" \
  --out_png "./data/ml/feature_corr_with_targets.png" \
  --include_targets
```
