"""Phase 10c — Relative conformal: calibrate |resid|/pred thay vì |resid| tuyệt đối.
Lý do: drift giá 2022→2023 làm width tuyệt đối từ 2022 bị thiếu ở 2023; width tương đối
tự co giãn theo segment và theo lạm phát. Fallback nhóm thưa -> nhóm gần nhất (không global).
Mục tiêu: coverage ~90% đều hơn giữa các segment. Eval trên val 2023, test 2024 chưa đụng.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RNG = 42
BEST_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=127,
                   min_child_samples=100, colsample_bytree=0.7, subsample=1.0, subsample_freq=1)
M_SMOOTH = 10
ALPHA = 0.9
SEG_EDGES = [0, 300_000, 500_000, 700_000, 1_000_000, np.inf]
SEG_NAMES = ['<$300k', '$300–500k', '$500–700k', '$700k–1M', '>$1M']

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2
NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = CAT + NUM + ['street_te']

def make_te(fit_df, apply_dfs, m=M_SMOOTH):
    prior = fit_df['resale_price'].mean()
    stats = fit_df.groupby('street_name')['resale_price'].agg(['mean', 'size'])
    town_mean = fit_df.groupby('town')['resale_price'].mean()
    def te(street, town):
        if street in stats.index:
            st = stats.loc[street]
            return (st['size'] * st['mean'] + m * prior) / (st['size'] + m)
        return town_mean.get(town, prior)
    for d in apply_dfs:
        d['street_te'] = [te(s, t) for s, t in zip(d['street_name'], d['town'])]

def fit_predict(fit_df, pred_dfs):
    make_te(fit_df, [fit_df] + pred_dfs)
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM + ['street_te']),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas'), CAT),
    ])
    model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
    model.fit(pre.fit_transform(fit_df[FEATURES]), fit_df['resale_price'].values)
    return [model.predict(pre.transform(d[FEATURES])) for d in pred_dfs]

def bucket_of(values):
    return pd.cut(values, bins=SEG_EDGES, labels=SEG_NAMES, right=False)

cal_df = df[df['year'] == 2022].copy()
val_df = df[df['year'] == 2023].copy()

# 1) calibration trên 2022 (model fit ≤2021): score tương đối |resid|/pred
[pred_cal] = fit_predict(df[df['year'] <= 2021].copy(), [cal_df])
scores = np.abs(cal_df['resale_price'].values - pred_cal) / pred_cal
cal_buckets = bucket_of(pred_cal)

# width tương đối theo nhóm; nhóm thưa -> dùng width nhóm GẦN NHẤT có đủ mẫu
rel_w = {}
order = SEG_NAMES
for i, g in enumerate(order):
    msk = cal_buckets == g
    if msk.sum() >= 300:
        rel_w[g] = (float(np.quantile(scores[msk], ALPHA)), int(msk.sum()), 'group')
# lan truyền fallback từ nhóm gần nhất có đủ mẫu
for i, g in enumerate(order):
    if g not in rel_w:
        for j in range(1, len(order)):
            for k in (i - j, i + j):
                if 0 <= k < len(order) and order[k] in rel_w:
                    rel_w[g] = (rel_w[order[k]][0], rel_w[order[k]][1], f'fallback←{order[k]}')
                    break
            if g in rel_w:
                break
print("Relative width theo nhóm giá dự đoán (calibration 2022):")
for g in order:
    w, n, src = rel_w[g]
    print(f"  {g:12s} n={n:5d}  w=±{w*100:5.1f}%  ({src})")

# 2) eval trên val 2023 (model refit ≤2022)
[pred_val] = fit_predict(df[df['year'] <= 2022].copy(), [val_df])
y_val = val_df['resale_price'].values
w_val = np.array([rel_w[b][0] * p for b, p in zip(bucket_of(pred_val), pred_val)])
covered = (y_val >= pred_val - w_val) & (y_val <= pred_val + w_val)
print(f"\n=== Coverage [pred·(1±w)] trên val 2023: {covered.mean()*100:.1f}% (mục tiêu 90%) ===")

rows = []
for seg, lo_p, hi_p in zip(SEG_NAMES, SEG_EDGES[:-1], SEG_EDGES[1:]):
    msk = (y_val >= lo_p) & (y_val < hi_p)
    if msk.sum() == 0:
        continue
    rows.append({'Segment (giá thật)': seg, 'n': int(msk.sum()),
                 'Coverage_pct': float(covered[msk].mean() * 100),
                 'Mean_width': float(w_val[msk].mean())})
seg_df = pd.DataFrame(rows)
print(seg_df.to_string(index=False))
seg_df.to_csv('v2_experiments/phase10c_rel_conformal.csv', index=False)
print("\nĐã lưu v2_experiments/phase10c_rel_conformal.csv")
