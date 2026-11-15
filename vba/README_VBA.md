# MarketLens VBA module: turning `MarketLens.xlsx` into `MarketLens.xlsm`

`MarketLens.bas` adds two things to the workbook that Python exports:

| What | How you use it |
|---|---|
| `BS_PRICE(S, K, T, r, sigma, [type], [q])` | Worksheet formula, e.g. `=BS_PRICE(100,100,1,5%,20%,"call")` returns `10.4506` |
| `BS_GREEK(greek, S, K, T, r, sigma, [type], [q])` | `greek` is `"delta"`, `"gamma"`, `"vega"`, `"theta"`, `"rho"` or `"price"` |
| `BS_IMPLIED_VOL(price, S, K, T, r, [type], [q])` | Newton-Raphson with a bisection fallback. Returns `#NUM!` if no volatility can produce the price |
| `RefreshVaR` macro | Reads the yellow inputs on **VaR Macro**, recalculates historical and parametric VaR/ES from the **Returns** sheet, fills the green table and rebuilds two charts |
| `AddRefreshButton` macro | Run it once to put a **Refresh VaR** button on the sheet |

Units: vega is per 1 volatility point, theta per calendar day, rho per 1% move in rates. These are the same conventions as `marketlens/options.py`.

## 1. Build the workbook

```bash
python -m marketlens.excel_export
```

This writes `reports/MarketLens.xlsx`. A plain `.xlsx` file cannot contain macros, which is why the next steps save it as `.xlsm`.

## 2. Show the Developer tab (one-time)

**File → Options → Customize Ribbon**, tick **Developer** on the right, then click **OK**.

## 3. Import the module

1. Open `reports/MarketLens.xlsx` in Excel.
2. Press **Alt + F11** to open the VBA editor.
3. In the **Project** pane, right-click **VBAProject (MarketLens.xlsx)** and choose **Import File…**.
4. Pick `vba/MarketLens.bas`. A module called **MarketLens** appears under **Modules**.
5. Optional check: **Debug → Compile VBAProject**. Nothing should happen, which means it compiled without errors.
6. Close the editor (**Alt + Q**).

## 4. Save as a macro-enabled workbook

**File → Save As**, set **Save as type** to **Excel Macro-Enabled Workbook (\*.xlsm)**, and save as `MarketLens.xlsm`.

When you reopen it, click **Enable Content** on the yellow security bar. Macros only run in files you trust, and this one is yours.

## 5. Try it

**Pricer sheet**
- Press **Ctrl + Alt + F9** to force a full recalculation.
- Column D (VBA) should now show numbers instead of `#NAME?`.
- Column E (VBA minus Excel formula) should be about 0, for example `1E-15`.
- `D25` (implied vol of a 10.4506 call) should show **20.00%**.

**VaR Macro sheet**
1. Run the button helper once: **Developer → Macros** (or **Alt + F8**) → `AddRefreshButton` → **Run**.
2. Click **Refresh VaR**. The green table fills in, and two charts appear: VaR/ES by method, and a histogram of daily portfolio returns.
3. With the default inputs, the **Diff VaR (bp)** and **Diff ES (bp)** columns should read about **0.00**. That shows the VBA matches the Python results.

    | Expected value with the default inputs | Historical | Parametric |
    |---|---|---|
    | 95% VaR | 1.711% | 1.810% |
    | 99% VaR | 2.471% | 2.549% |

4. Now change something: set TCS to 40% and INFY to 0%, or the window to 500 days, or confidence level 2 to 97.5%. Click **Refresh VaR** again. The Python reference columns stay fixed, so the difference now shows how much your change moved the risk.

## How RefreshVaR works

1. Reads the 5 weights, the window, the portfolio value and the 2 confidence levels.
2. Finds each stock's column on the **Returns** sheet by name and loads the last `window` rows into an array in one go (much faster than reading cell by cell).
3. Portfolio return each day = Σ weight × stock return.
4. **Historical:** VaR = −`PERCENTILE.INC`(returns, 1 − c). ES = −(average of the returns at or below that percentile).
5. **Parametric:** VaR = z·σ − μ and ES = σ·φ(z)/(1 − c) − μ, with z = `NORM.S.INV(c)`. σ of the portfolio return series is exactly √(wᵀΣw), so no covariance matrix is needed.
6. Writes the table, bins the returns into 30 buckets for the histogram, and rebuilds both charts. Old charts are deleted first, so running it twice never stacks duplicates.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `#NAME?` in Pricer column D | The module isn't imported, or the file was saved as `.xlsx`. Re-import, save as `.xlsm`, then press Ctrl + Alt + F9 |
| "Macros have been disabled" | Close and reopen, then click **Enable Content**. Or use **File → Info → Enable Content** |
| "Stock 'X' not found…" | A name in `VaR Macro!B5:B9` doesn't match a header in row 1 of **Returns** |
| Weight warning | The weights don't sum to 100%. You can continue anyway, but the VaR is then for a leveraged or under-invested portfolio |
| Compile error on `Norm_S_Dist` | Needs Excel 2010 or later |
