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
- auto serial block traversal
- fallback block selection by valid numeric points
- multi-start fitting
- frequency filtering / high-frequency point dropping
- fit quality export (`json` + residual `csv`)

### CLI

```bash
python src/ecm_fit.py \
  --xlsx <xlsx_path> \
  [--sheet 02_PreEIS] \
  [--block 2] \
  [--serial <serial>] \
  [--circuit "R0-p(R1,CPE1)-p(R2,CPE2)"] \
  [--guess ""] \
  [--fmin <hz>] [--fmax <hz>] \
  [--drop_first_n <n>] \
  [--n_starts <n>] \
  [--weight_by_modulus] \
  --out_dir <output_dir>
```

### Common Parameters

- `--sheet`: target sheet (default `02_PreEIS`)
- `--block`: preferred block index; script can fallback to best block
- `--serial`: only run one serial, otherwise run all detected serials
- `--circuit`: ECM topology
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

### Output

```text
data/test_ecm/<serial>/nyquist_fit__<sheet>__block<k>.png
data/test_ecm/<serial>/fit_metrics__<sheet>__block<k>.json
data/test_ecm/<serial>/fit_residuals__<sheet>__block<k>.csv
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
