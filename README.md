# HDB Resale Price Prediction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-66CC33)](https://lightgbm.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

Predicting Singapore HDB resale flat prices from **221k transaction records (2015–2024)** — with a
methodologically rigorous, time-aware evaluation protocol and **statistically calibrated prediction
intervals**.

The interesting part of this project is not the model. It is the **evaluation discipline**: the
original version of this project used a random train/test split on temporal data and reported
R² = 0.945. This repo explains why that number was optimistic, rebuilds the pipeline with a strict
time-based protocol, and reports what the *honest* number looks like — then closes the gap with
better features and calibrated uncertainty.

## Headline results — test year 2024 (never seen during training or tuning)

| | v1 — random split *(flawed)* | v2 — time split | v3 — v2 + street TE + conformal |
|---|---|---|---|
| R² | 0.9452 *(optimistic)* | 0.9008 | **0.9217** |
| RMSE | $39,955 | $58,478 | **$51,963** |
| MAE | $28,296 | $41,180 | **$37,175** |
| Predictions within ±10% | 84.05% | 79.46% | **84.25%** |
| Output | point estimate | point estimate | estimate + **90% interval** (92.9% empirical coverage) |
| Model size | ~250 MB | < 1 MB | < 1 MB |

**Why the v1 number was inflated.** Resale prices are non-stationary: the median price rose ~33%
between the 2015–2022 training window and 2024. A random split lets the model learn the price
distribution of the very years it is evaluated on. v1's own 5-fold cross-validation averaged
R² = 0.881 — the gap to 0.945 is a direct measure of the split-induced optimism. The v3 result
(0.9217) nearly matches the flawed number *on genuinely unseen years*, using a protocol that
survives scrutiny.

## Evaluation protocol

```
Data audit → time split → baselines → feature ablation → target experiments
→ hyperparameter tuning (year-based CV) → final model → conformal calibration → error analysis
```

- **Time-based split**: train 2015–2022 (178,587) / validation 2023 (25,478) / test 2024 (16,906).
  The test year is touched exactly once per model version, at the very end.
- **All decisions** — features, transforms, hyperparameters, model family — are made on the shared
  2023 validation set. Nothing is selected on test.
- **Year-explicit cross-validation** for tuning: (2015–2019→2020), (2015–2020→2021),
  (2015–2021→2022). No `TimeSeriesSplit` guesswork; folds are readable in a report.
- **Baselines first**: a median predictor scores R² = −0.62 on 2023 — direct evidence of the
  distribution shift that makes naive evaluation misleading.

## Modeling decisions (each backed by an experiment)

| Decision | Evidence |
|---|---|
| Model: **LightGBM** over RandomForest | Ties RF on accuracy, ~300× faster to fit (0.9s vs 305s), <1 MB vs 250 MB |
| Add **`storey_mid`** (midpoint floor level) | −$4,290 RMSE vs. baseline features (ablation, val 2023) |
| Encode storey as **numeric mid-floor**, not one-hot | Better generalization on rare ranges (floors 40+ have <100 rows each) |
| **Drop `distance_from_expressway`** | ΔRMSE ≈ 0 once storey is included; it was absorbing noise |
| Add **smoothed target encoding of `street_name`** | −$5,006 RMSE on val; 3 leakage safeguards (train-only fit, smoothing m=10, town-level fallback for unseen streets — only 12/25,478 val rows fall back) |
| Keep **raw target** (no log transform) | log1p is worse on every metric for both LightGBM and RF |
| Tuning: wide search **with `max_features`** | v1's search space was centered on already-known best values — tuning in name only |

### Prediction intervals: what didn't work, and what did

- ❌ **Vanilla quantile regression** (LightGBM, α = 0.1/0.5/0.9): empirical coverage **53.5%**
  against an 80% target — and 16.9% on the >$1M segment. A single global quantile model cannot
  capture segment-dependent heteroscedasticity.
- ✅ **Relative conformal calibration**: conformal scores `|residual| / prediction`, computed on a
  held-out calibration year (2023, model fit ≤2022), bucketed by *predicted* price (inference never
  sees true prices). Empirical coverage on the untouched 2024 test year: **92.9%** at a 90% target.

## Known limitations (measured, not hidden)

- **>$1M segment: 66.9% interval coverage.** The calibration year contained no transactions in
  this bucket, so its width falls back to the neighboring segment. Conformal guarantees *marginal*
  coverage; segment-conditional guarantees need more calibration data at the high end.
- **Point accuracy degrades with price**: 86.8% within ±10% in the $300–500k segment vs. ~60% in
  expensive central towns (Serangoon, Geylang, Kallang/Whampoa). This is where the model needs
  location features it does not currently have.
- **No micro-location features yet** (distance to MRT/CBD, `flat_model`, `month`) — the dataset
  lacks them; adding the official data.gov.sg fields is the next step.

## Repository structure

```
ML_Cki_v2.ipynb        # main notebook — 8 sections, runs end-to-end in ~2 minutes
v2_experiments/        # full provenance: script + result CSVs for every experiment (phases 0–11)
SUMMARY_v1_vs_v2.md    # 3-version comparison + interview Q&A notes
requirements.txt
```

Every number in this README is reproducible from the scripts in `v2_experiments/`.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook ML_Cki_v2.ipynb
```

Inference with the trained bundle (preprocessor + point model + conformal widths; not committed —
retrain via the notebook or `v2_experiments/phase11_final_v3.py`):

```python
import joblib
import pandas as pd

bundle = joblib.load("resale_price_model_v3.joblib")

flat = pd.DataFrame([{
    "town": "BUKIT BATOK", "flat_type": "4 ROOM", "year": 2024,
    "floor_area_sqm": 82.0, "remaining_lease_years": 68,
    "storey_range": "07 TO 09", "street_name": "BUKIT BATOK STREET 32",
}])

lo_hi = flat["storey_range"].str.extract(r"(\d+)\s*TO\s*(\d+)")
flat["storey_mid"] = (lo_hi[0].astype(int) + lo_hi[1].astype(int)) / 2

te = bundle["street_te"]
name = flat["street_name"][0]
if name in te["stats"].index:
    s = te["stats"].loc[name]
    flat["street_te"] = (s["size"] * s["mean"] + te["m"] * te["prior"]) / (s["size"] + te["m"])
else:
    flat["street_te"] = te["town_mean"].get(flat["town"][0], te["prior"])

cols = ["town", "flat_type", "year", "floor_area_sqm",
        "remaining_lease_years", "storey_mid", "street_te"]
pred = bundle["point_model"].predict(bundle["preprocessor"].transform(flat[cols]))[0]
w = bundle["rel_widths"]["$300–500k"]

print(f"Predicted: ${pred:,.0f}")
print(f"90% interval: ${pred - w * pred:,.0f} – ${pred + w * pred:,.0f}")
```

## Roadmap

1. **Micro-location features** — geocode block/street, compute distance to nearest MRT and CBD
   (the most promising lever for the high-price segment and central towns)
2. **Official data.gov.sg fields** — `flat_model` and `month`, enabling a rolling price index and
   quarterly walk-forward validation
3. **Segment-conditional conformal calibration** for the >$1M bucket

---

*Data: Singapore HDB resale transactions, 2015–2024. Built with pandas, scikit-learn, and LightGBM.*
