"""Phase 9 — Target encoding `street_name` (566 giá trị), chống leakage 3 lớp:
  1. Encoding fit CHỈ trên train (≤2022) khi đánh giá trên val 2023.
  2. Smoothed encoding: street ít mẫu blend về prior, weight m thí nghiệm {10, 50, 200}.
  3. Street lạ ở val → fallback town-level (không fallback về global trừ khi cả town lạ).

Hai biến thể:
  A. TE thô trên giá: street_mean giá (mô hình có `year` nên tự bù xu hướng).
  B. Street premium index: encode mean(price / median[town×flat_type×năm_trước]),
     trong đó median chỉ dùng dữ liệu CÁC NĂM TRƯỚC (point-in-time safe).
Đánh giá trên val 2023 bằng LightGBM đã tuned (Phase 5).
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
BEST_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=127,
                   min_child_samples=100, colsample_bytree=0.7, subsample=1.0, subsample_freq=1)

def within_pct(y_true, y_pred, tol):
    return float(((y_pred >= y_true * (1 - tol)) & (y_pred <= y_true * (1 + tol))).mean() * 100)

def evaluate(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'Pct_within_10pct': within_pct(y_true, y_pred, 0.10)}

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']

# ---- Biến thể B: rel_price chuẩn hóa theo town×flat_type×năm TRƯỚC (point-in-time) ----
grp = df.groupby(['town', 'flat_type', 'year'])['resale_price'].median().rename('grp_med').reset_index()
grp['grp_med_prev'] = grp.groupby(['town', 'flat_type'])['grp_med'].shift(1)
df = df.merge(grp[['town', 'flat_type', 'year', 'grp_med_prev']], on=['town', 'flat_type', 'year'], how='left')
# năm đầu tiên của mỗi nhóm (không có năm trước) -> dùng median toàn cục của train làm mốc
global_med_2015 = df.loc[df['year'] <= 2022, 'resale_price'].median()
df['grp_med_prev'] = df['grp_med_prev'].fillna(global_med_2015)
df['rel_price'] = df['resale_price'] / df['grp_med_prev']

train = df[df['year'] <= 2022].copy()
val = df[df['year'] == 2023].copy()
y_val = val['resale_price'].values

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas'), CAT),
])
X_val_base = pre.fit(train[CAT + NUM]).transform(val[CAT + NUM])
X_train_base = pre.transform(train[CAT + NUM])
y_train = train['resale_price'].values

prior_rel = train['rel_price'].mean()
town_rel = train.groupby('town')['rel_price'].mean()
street_stats_A = train.groupby('street_name')['resale_price'].agg(['mean', 'size'])
street_stats_B = train.groupby('street_name')['rel_price'].agg(['mean', 'size'])
prior_A = train['resale_price'].mean()
town_A = train.groupby('town')['resale_price'].mean()

def te_A(row_street, row_town, m):
    st = street_stats_A.loc[row_street] if row_street in street_stats_A.index else None
    if st is None:
        return town_A.get(row_town, prior_A)
    return (st['size'] * st['mean'] + m * prior_A) / (st['size'] + m)

def te_B(row_street, row_town, m):
    st = street_stats_B.loc[row_street] if row_street in street_stats_B.index else None
    if st is None:
        return town_rel.get(row_town, prior_rel)
    return (st['size'] * st['mean'] + m * prior_rel) / (st['size'] + m)

rows = []
# Control: không TE
m0 = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
m0.fit(X_train_base, y_train)
r = evaluate(y_val, m0.predict(X_val_base)); r['Config'] = 'control (6 features, no TE)'
rows.append(r)
print(f"{r['Config']:42s} R2={r['R2']:.4f} RMSE={r['RMSE']:>10,.0f} MAE={r['MAE']:>9,.0f} ±10%={r['Pct_within_10pct']:5.1f}%")

for variant, te_fn, prior in [('A: raw-price TE', te_A, prior_A), ('B: premium TE (point-in-time)', te_B, prior_rel)]:
    for m in [10, 50, 200]:
        t0 = time.time()
        te_tr = np.array([te_fn(s, t, m) for s, t in zip(train['street_name'], train['town'])]).reshape(-1, 1)
        te_va = np.array([te_fn(s, t, m) for s, t in zip(val['street_name'], val['town'])]).reshape(-1, 1)
        model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
        model.fit(np.hstack([X_train_base, te_tr]), y_train)
        r = evaluate(y_val, model.predict(np.hstack([X_val_base, te_va])))
        r['Config'] = f'{variant}, m={m}'
        rows.append(r)
        print(f"{r['Config']:42s} R2={r['R2']:.4f} RMSE={r['RMSE']:>10,.0f} MAE={r['MAE']:>9,.0f} ±10%={r['Pct_within_10pct']:5.1f}% ({time.time()-t0:.0f}s)")

out = pd.DataFrame(rows)[['Config', 'R2', 'RMSE', 'MAE', 'Pct_within_10pct']]
out.to_csv('v2_experiments/phase9_street_te.csv', index=False)
n_unseen = (~val['street_name'].isin(street_stats_A.index)).sum()
print(f"\nStreet lạ trong val 2023: {n_unseen}/{len(val)} dòng (fallback town-level)")
print("Đã lưu v2_experiments/phase9_street_te.csv")
