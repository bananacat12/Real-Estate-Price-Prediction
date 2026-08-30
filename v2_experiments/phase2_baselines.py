"""Phase 2 — Baselines trên time-based split.
Train 2015-2022 | Validation 2023 (mọi so sánh dùng val) | Test 2024: KHÔNG đụng đến.
Features: base (year, town, flat_type, floor_area_sqm, remaining_lease_years, distance_ord)
— trùng với bộ features của notebook v1 để so sánh trực tiếp.
"""
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

RNG = 42

def within_pct(y_true, y_pred, tol):
    lo, hi = y_true * (1 - tol), y_true * (1 + tol)
    return float(((y_pred >= lo) & (y_pred <= hi)).mean() * 100)

def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'Pct_within_5pct': within_pct(y_true, y_pred, 0.05),
        'Pct_within_10pct': within_pct(y_true, y_pred, 0.10),
    }

def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), NUM),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]), CAT),
    ])

df = pd.read_csv('resale.csv')
# distance: 6 bậc có thứ tự -> mã hóa numeric theo midpoint, tài liệu hóa mapping
DIST_MAP = {'<=50m': 25, '51-100m': 75, '101-150m': 125, '151-300m': 225, '301-500m': 400, '>500m': 500}
df['distance_ord'] = df['distance_from_expressway'].map(DIST_MAP)
assert df['distance_ord'].notna().all()

CAT = ['town', 'flat_type']
NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'distance_ord']
FEATURES = NUM + CAT

train = df[df['year'] <= 2022]
val = df[df['year'] == 2023]
X_train, y_train = train[FEATURES], train['resale_price']
X_val, y_val = val[FEATURES], val['resale_price']

models = {
    'Median baseline': DummyRegressor(strategy='median'),
    'Ridge': Ridge(alpha=1.0, random_state=RNG),
    'RandomForest (default-ish)': RandomForestRegressor(n_estimators=100, random_state=RNG, n_jobs=-1),
    'LightGBM (default)': None,  # fill below
}
import lightgbm as lgb
models['LightGBM (default)'] = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1)

rows = []
for name, est in models.items():
    pipe = Pipeline([
        ('prep', make_preprocessor(scale_numeric=(name == 'Ridge'))),
        ('model', est),
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0
    m = evaluate(y_val.values, pipe.predict(X_val))
    m.update(Model=name, fit_seconds=round(fit_s, 1))
    rows.append(m)
    print(f"{name:32s} R2={m['R2']:.4f} RMSE={m['RMSE']:>10,.0f} MAE={m['MAE']:>9,.0f} "
          f"±5%={m['Pct_within_5pct']:5.1f}% ±10%={m['Pct_within_10pct']:5.1f}% ({fit_s:.0f}s)")

out = pd.DataFrame(rows)[['Model', 'R2', 'RMSE', 'MAE', 'Pct_within_5pct', 'Pct_within_10pct', 'fit_seconds']]
out.to_csv('v2_experiments/phase2_baselines.csv', index=False)
print("\nĐã lưu v2_experiments/phase2_baselines.csv")
print("\nTham chiếu v1 (random split, test 20%): R2=0.9452 RMSE=39,955 MAE=28,296 ±10%=84.05%")
