"""Phase 10b — Conformal prediction phân đoạn (Mondrian conformal) thay cho quantile thuần.
Protocol (không đụng test 2024):
  1. Fit point model (LGBM + street TE) trên ≤2021 -> residual trên 2022 (năm calibration riêng).
  2. Tính width w_g = quantile_90(|resid|) theo NHÓM GIÁ DỰ ĐOÁN (không dùng giá thật —
     vì lúc inference chỉ biết giá dự đoán), fallback global nếu nhóm ít mẫu.
  3. Refit model trên ≤2022 -> dự đoán val 2023, khoảng = pred ± w_g(nhóm theo giá dự đoán).
  4. Đánh giá coverage trên val 2023 (calibration set và eval set tách biệt).
So sánh với quantile thuần (Phase 10: coverage 53.5%).
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
ALPHA = 0.9  # mức phủ mục tiêu 90%
SEG_EDGES = [0, 300_000, 500_000, 700_000, 1_000_000, np.inf]
SEG_NAMES = ['<$300k', '$300–500k', '$500–700k', '$700k–1M', '>$1M']
MIN_CAL = 300  # nhóm calibration cần tối thiểu bao nhiêu mẫu, nếu ít hơn -> dùng global

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

def make_pre():
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM + ['street_te']),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas'), CAT),
    ])

def fit_model(fit_df, pred_dfs):
    make_te(fit_df, [fit_df] + pred_dfs)
    pre = make_pre()
    X = pre.fit_transform(fit_df[FEATURES])
    y = fit_df['resale_price'].values
    model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
    model.fit(X, y)
    preds = [model.predict(pre.transform(d[FEATURES])) for d in pred_dfs]
    return preds

cal_df = df[df['year'] == 2022].copy()
val_df = df[df['year'] == 2023].copy()

# 1) model fit ≤2021, residual 2022 (năm calibration riêng)
[pred_cal] = fit_model(df[df['year'] <= 2021].copy(), [cal_df])
resid_cal = cal_df['resale_price'].values - pred_cal
abs_resid = np.abs(resid_cal)
cal_buckets = pd.cut(pred_cal, bins=SEG_EDGES, labels=SEG_NAMES, right=False)

widths = {}
for g in SEG_NAMES:
    msk = cal_buckets == g
    widths[g] = float(np.quantile(abs_resid[msk], ALPHA)) if msk.sum() >= MIN_CAL else None
global_w = float(np.quantile(abs_resid, ALPHA))
for g in SEG_NAMES:
    if widths[g] is None:
        widths[g] = global_w
print("Width theo nhóm giá dự đoán (calibration 2022, mức phủ mục tiêu 90%):")
for g in SEG_NAMES:
    n_g = int((cal_buckets == g).sum())
    src = 'group' if (cal_buckets == g).sum() >= MIN_CAL else 'global-fallback'
    print(f"  {g:12s} n={n_g:5d}  w=±${widths[g]:>9,.0f}  ({src})")

# 2) refit ≤2022, dự đoán val 2023, khoảng = pred ± w(nhóm theo giá dự đoán)
[pred_val] = fit_model(df[df['year'] <= 2022].copy(), [val_df])
y_val = val_df['resale_price'].values
bucket_val = pd.cut(pred_val, bins=SEG_EDGES, labels=SEG_NAMES, right=False)
w_val = np.array([widths[b] for b in bucket_val])
lo, hi = pred_val - w_val, pred_val + w_val
covered = (y_val >= lo) & (y_val <= hi)
print(f"\n=== Coverage [pred±w] trên val 2023: {covered.mean()*100:.1f}% (mục tiêu 90%) ===")

rows = []
for seg, lo_p, hi_p in zip(SEG_NAMES, SEG_EDGES[:-1], SEG_EDGES[1:]):
    msk_true = (y_val >= lo_p) & (y_val < hi_p)
    if msk_true.sum() == 0:
        continue
    rows.append({'Segment (giá thật)': seg, 'n': int(msk_true.sum()),
                 'Coverage_pct': float(covered[msk_true].mean() * 100),
                 'Mean_width': float(w_val[msk_true].mean())})
seg_df = pd.DataFrame(rows)
print(seg_df.to_string(index=False))
seg_df.to_csv('v2_experiments/phase10b_conformal_segments.csv', index=False)
print("\nĐã lưu v2_experiments/phase10b_conformal_segments.csv")
