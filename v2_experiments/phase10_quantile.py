"""Phase 10 — Quantile regression (LGBM, alpha=0.1/0.5/0.9) trên feature set cuối:
6 features + street TE (biến thể A, m=10 — thắng Phase 9).
Đánh giá trên val 2023: độ phủ khoảng [q10, q90], độ rộng khoảng, MAE của q50.
Mọi quyết định trên val; test 2024 chưa đụng.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RNG = 42
BEST_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=127,
                   min_child_samples=100, colsample_bytree=0.7, subsample=1.0, subsample_freq=1)
M_SMOOTH = 10

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = CAT + NUM

train = df[df['year'] <= 2022].copy()
val = df[df['year'] == 2023].copy()

# Street TE (A, m=10) — fit chỉ trên train
prior = train['resale_price'].mean()
stats = train.groupby('street_name')['resale_price'].agg(['mean', 'size'])
town_mean = train.groupby('town')['resale_price'].mean()

def te(street, town, m=M_SMOOTH):
    if street in stats.index:
        st = stats.loc[street]
        return (st['size'] * st['mean'] + m * prior) / (st['size'] + m)
    return town_mean.get(town, prior)

for d in (train, val):
    d['street_te'] = [te(s, t) for s, t in zip(d['street_name'], d['town'])]
NUM_FINAL = NUM + ['street_te']
FEATURES = CAT + NUM_FINAL

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM_FINAL),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas'), CAT),
])
X_train = pre.fit_transform(train[FEATURES])
y_train = train['resale_price'].values
X_val = pre.transform(val[FEATURES])
y_val = val['resale_price'].values

preds = {}
for alpha in [0.1, 0.5, 0.9]:
    m = lgb.LGBMRegressor(objective='quantile', alpha=alpha, random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
    m.fit(X_train, y_train)
    preds[alpha] = m.predict(X_val)

q10, q50, q90 = preds[0.1], preds[0.5], preds[0.9]
covered = (y_val >= q10) & (y_val <= q90)
print(f"Coverage [q10,q90] toàn val: {covered.mean()*100:.1f}% (kỳ vọng lý thuyết ~80%)")
print(f"Độ rộng khoảng trung bình: ${(q90-q10).mean():,.0f} | median: ${np.median(q90-q10):,.0f}")
print(f"MAE của q50: ${mean_absolute_error(y_val, q50):,.0f} (so với mean-model $39,774 ở Phase 9)")

# MAE q50 so với mean model: fit mean model cùng feature set
mm = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
mm.fit(X_train, y_train)
print(f"MAE mean model (cùng features): ${mean_absolute_error(y_val, mm.predict(X_val)):,.0f}")

# Coverage + width theo segment
segs = [(0, 300_000, '<$300k'), (300_000, 500_000, '$300–500k'), (500_000, 700_000, '$500–700k'),
        (700_000, 1_000_000, '$700k–1M'), (1_000_000, np.inf, '>$1M')]
rows = []
for lo, hi, name in segs:
    msk = (y_val >= lo) & (y_val < hi)
    if msk.sum() == 0:
        continue
    rows.append({'Segment': name, 'n': int(msk.sum()),
                 'Coverage_pct': float(covered[msk].mean() * 100),
                 'Mean_width': float((q90 - q10)[msk].mean()),
                 'MAE_q50': float(np.abs(y_val[msk] - q50[msk]).mean())})
seg_df = pd.DataFrame(rows)
print("\n=== Coverage & độ rộng khoảng theo segment (val 2023) ===")
print(seg_df.to_string(index=False))
seg_df.to_csv('v2_experiments/phase10_quantile_segments.csv', index=False)
print("\nĐã lưu v2_experiments/phase10_quantile_segments.csv")
