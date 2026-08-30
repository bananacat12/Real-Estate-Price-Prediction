"""Phase 4 — Raw target vs log1p(target).
Feature set thắng Phase 3: year + floor_area + lease + distance?KHÔNG + storey_mid.
Metrics tính ở KHÔNG GIAN GỐC (giá SGD) cho cả hai — log chỉ là transform khi fit.
Chạy cả LightGBM và RF để kết luận không phụ thuộc 1 model family.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
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
        'Pct_within_5pct': within_pct(y_true, y_pred, 0.05),
        'Pct_within_10pct': within_pct(y_true, y_pred, 0.10),
    }

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = NUM + CAT

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]).set_output(transform='pandas'), CAT),
])

train = df[df['year'] <= 2022]
val = df[df['year'] == 2023]
X_train = pre.fit_transform(train[FEATURES])
X_val = pre.transform(val[FEATURES])
y_val = val['resale_price'].values

rows = []
for model_name, mk in [
    ('LightGBM', lambda: lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1)),
    ('RandomForest', lambda: RandomForestRegressor(n_estimators=100, random_state=RNG, n_jobs=-1)),
]:
    for target_name, tfwd, tinv in [('raw', lambda y: y, lambda p: p),
                                     ('log1p', np.log1p, np.expm1)]:
        m = mk()
        t0 = time.time()
        m.fit(X_train, tfwd(train['resale_price']))
        fit_s = time.time() - t0
        pred = tinv(m.predict(X_val))
        r = evaluate(y_val, pred)
        r.update(Model=f'{model_name} [{target_name}]', fit_seconds=round(fit_s, 1))
        rows.append(r)
        print(f"{r['Model']:26s} R2={r['R2']:.4f} RMSE={r['RMSE']:>10,.0f} MAE={r['MAE']:>9,.0f} "
              f"±5%={r['Pct_within_5pct']:5.1f}% ±10%={r['Pct_within_10pct']:5.1f}% ({fit_s:.0f}s)")

out = pd.DataFrame(rows)[['Model', 'R2', 'RMSE', 'MAE', 'Pct_within_5pct', 'Pct_within_10pct', 'fit_seconds']]
out.to_csv('v2_experiments/phase4_target.csv', index=False)
print("\nĐã lưu v2_experiments/phase4_target.csv")
