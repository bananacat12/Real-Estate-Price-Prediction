"""Phase 7 — Phân tích sau final-evaluation (chỉ báo cáo, không ảnh hưởng quyết định model):
  1. Residual analysis (charts)
  2. Performance theo price segment
  3. Performance theo town
  4. Permutation importance Ở MỨC FEATURE (aggregate sau one-hot), trên test 2024
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RNG = 42
OUT = 'v2_experiments'

preds = pd.read_csv(f'{OUT}/phase6_test2024_predictions.csv')
y_true = preds['resale_price'].values
y_pred = preds['predicted'].values
resid = y_true - y_pred

def within(y, p, tol):
    return ((p >= y * (1 - tol)) & (p <= y * (1 + tol)))

# --- 1. Residual charts ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_pred, resid, alpha=0.3, s=6)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted price (SGD)'); ax.set_ylabel('Residual (actual - predicted)')
ax.set_title('Residuals vs Predicted — test 2024 (v2)')
plt.tight_layout(); plt.savefig(f'{OUT}/v2_residuals_vs_predicted.png', dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(resid, bins=60, color='purple', alpha=0.75)
ax.set_xlabel('Residual (SGD)'); ax.set_title('Residual distribution — test 2024 (v2)')
plt.tight_layout(); plt.savefig(f'{OUT}/v2_residuals_distribution.png', dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_true, y_pred, alpha=0.3, s=6)
lims = [y_true.min(), y_true.max()]
ax.plot(lims, lims, 'r--')
ax.set_xlabel('Actual price (SGD)'); ax.set_ylabel('Predicted price (SGD)')
ax.set_title('Actual vs Predicted — test 2024 (v2)')
plt.tight_layout(); plt.savefig(f'{OUT}/v2_actual_vs_predicted.png', dpi=110); plt.close()

# --- 2. Price segments ---
segments = [(0, 300_000, '<$300k'), (300_000, 500_000, '$300–500k'),
            (500_000, 700_000, '$500–700k'), (700_000, 1_000_000, '$700k–1M'),
            (1_000_000, np.inf, '>$1M')]
rows = []
for lo, hi, name in segments:
    m = (y_true >= lo) & (y_true < hi)
    if m.sum() == 0:
        continue
    rows.append({
        'Segment': name, 'n': int(m.sum()),
        'MAE': float(np.abs(resid[m]).mean()),
        'RMSE': float(np.sqrt((resid[m] ** 2).mean())),
        'Pct_within_10pct': within(y_true[m], y_pred[m], 0.10).mean() * 100,
    })
seg_df = pd.DataFrame(rows)
print('=== Performance theo price segment (test 2024) ===')
print(seg_df.to_string(index=False))
seg_df.to_csv(f'{OUT}/phase7_price_segments.csv', index=False)

# --- 3. By town ---
town_rows = []
for town, g in preds.groupby('town'):
    r = g['resale_price'].values; p = g['predicted'].values
    town_rows.append({'Town': town, 'n': len(g),
                      'MAE': float(np.abs(r - p).mean()),
                      'Pct_within_10pct': within(r, p, 0.10).mean() * 100})
town_df = pd.DataFrame(town_rows).sort_values('Pct_within_10pct', ascending=False)
print('\n=== ±10% theo town (test 2024) — top 5 / bottom 5 ===')
print(town_df.head(5).to_string(index=False)); print('...'); print(town_df.tail(5).to_string(index=False))
town_df.to_csv(f'{OUT}/phase7_by_town.csv', index=False)

# --- 4. Permutation importance ở mức FEATURE ---
df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2
NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = NUM + CAT

pipe = joblib.load('resale_price_model_v2.joblib')
test_df = df[df['year'] == 2024][FEATURES + ['resale_price']].reset_index(drop=True)
assert len(test_df) == len(preds)

base_rmse = float(np.sqrt((resid ** 2).mean()))
rng = np.random.default_rng(RNG)
imp_rows = []
for feat in FEATURES:
    deltas = []
    for _ in range(5):
        shuffled = test_df.copy()
        shuffled[feat] = rng.permutation(shuffled[feat].values)
        p = pipe.predict(shuffled[FEATURES])
        deltas.append(np.sqrt(((shuffled['resale_price'].values - p) ** 2).mean()) - base_rmse)
    imp_rows.append({'Feature': feat, 'dRMSE_mean': float(np.mean(deltas)),
                     'dRMSE_std': float(np.std(deltas))})
imp_df = pd.DataFrame(imp_rows).sort_values('dRMSE_mean', ascending=False)
print('\n=== Permutation importance (mức feature, ΔRMSE khi xáo trộn; test 2024, 5 repeats) ===')
print(imp_df.to_string(index=False))
imp_df.to_csv(f'{OUT}/phase7_permutation_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df['Feature'], imp_df['dRMSE_mean'], xerr=imp_df['dRMSE_std'], color='steelblue')
ax.invert_yaxis(); ax.set_xlabel('ΔRMSE khi xáo trộn feature (SGD)')
ax.set_title('Permutation importance (feature-level) — test 2024')
plt.tight_layout(); plt.savefig(f'{OUT}/v2_permutation_importance.png', dpi=110); plt.close()

print('\nĐã lưu charts + 3 CSV vào v2_experiments/')
