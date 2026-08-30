"""Phase 3 — Feature ablation trên LightGBM (fit nhanh ~1s, độ chính xác ngang RF ở Phase 2).
Cùng đánh giá trên validation 2023. Thiết kế A/B/C/D như đã duyệt:
  A: year + town + flat_type + floor_area + lease          (base không distance, không storey)
  B: A + distance                                          (trùng bộ features v1)
  C1/C2: A + storey (onehot / mid-floor numeric)
  D1/D2: A + distance + storey (onehot / mid-floor numeric)
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RNG = 42

def within_pct(y_true, y_pred, tol):
    lo, hi = y_true * (1 - tol), y_true * (1 + tol)
    return float(((y_pred >= lo) & (y_pred <= hi)).mean() * 100)

def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'Pct_within_10pct': within_pct(y_true, y_pred, 0.10),
    }

df = pd.read_csv('resale.csv')
DIST_MAP = {'<=50m': 25, '51-100m': 75, '101-150m': 125, '151-300m': 225, '301-500m': 400, '>500m': 500}
df['distance_ord'] = df['distance_from_expressway'].map(DIST_MAP)

# mid-floor: '04 TO 06' -> 5.0
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2
assert df['storey_mid'].notna().all()

NUM_BASE = ['year', 'floor_area_sqm', 'remaining_lease_years']
CAT_BASE = ['town', 'flat_type']

def make_pipe(num_feats, cat_feats):
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num_feats),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), cat_feats),
    ])
    return Pipeline([('prep', pre),
                     ('model', lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1))])

train = df[df['year'] <= 2022]
val = df[df['year'] == 2023]
y_val = val['resale_price'].values

configs = {
    'A  base (5 feats)':            (NUM_BASE, CAT_BASE),
    'B  A + distance':              (NUM_BASE + ['distance_ord'], CAT_BASE),
    'C1 A + storey (onehot)':       (NUM_BASE, CAT_BASE + ['storey_range']),
    'C2 A + storey (mid-floor)':    (NUM_BASE + ['storey_mid'], CAT_BASE),
    'D1 A + dist + storey (ohe)':   (NUM_BASE + ['distance_ord'], CAT_BASE + ['storey_range']),
    'D2 A + dist + storey (mid)':   (NUM_BASE + ['distance_ord', 'storey_mid'], CAT_BASE),
}

rows = []
for name, (num_feats, cat_feats) in configs.items():
    pipe = make_pipe(num_feats, cat_feats)
    t0 = time.time()
    pipe.fit(train[num_feats + cat_feats], train['resale_price'])
    fit_s = time.time() - t0
    m = evaluate(y_val, pipe.predict(val[num_feats + cat_feats]))
    m.update(Config=name, fit_seconds=round(fit_s, 1))
    rows.append(m)
    print(f"{name:30s} R2={m['R2']:.4f} RMSE={m['RMSE']:>10,.0f} MAE={m['MAE']:>9,.0f} ±10%={m['Pct_within_10pct']:5.1f}% ({fit_s:.1f}s)")

out = pd.DataFrame(rows)[['Config', 'R2', 'RMSE', 'MAE', 'Pct_within_10pct', 'fit_seconds']]
out.to_csv('v2_experiments/phase3_ablation.csv', index=False)
print("\nĐã lưu v2_experiments/phase3_ablation.csv")
