"""Phase 11 — FINAL v3: point model (6 features + street TE) + relative conformal intervals.
Protocol:
  - Widths calibrate trên residual năm 2023 (từ model fit ≤2022 — năm held-out liền trước test).
  - Point model cuối: fit ≤2023, dự đoán test 2024 DUY NHẤT MỘT LẦN.
  - Tất cả quyết định (TE m=10, conformal relative) đã chốt trên val trước; test chỉ báo cáo.
Lưu bundle v3: preprocessor + point model + widths + edges.
"""
import joblib
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
M_SMOOTH = 10
ALPHA = 0.9
SEG_EDGES = [0, 300_000, 500_000, 700_000, 1_000_000, np.inf]
SEG_NAMES = ['<$300k', '$300–500k', '$500–700k', '$700k–1M', '>$1M']

def within_pct(y_true, y_pred, tol):
    return float(((y_pred >= y_true * (1 - tol)) & (y_pred <= y_true * (1 + tol))).mean() * 100)

def evaluate(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'Pct_within_5pct': within_pct(y_true, y_pred, 0.05),
            'Pct_within_10pct': within_pct(y_true, y_pred, 0.10)}

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2
NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = CAT + NUM + ['street_te']

def stats_of(f):
    return f.groupby('street_name')['resale_price'].agg(['mean', 'size'])

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

def bucket_of(values):
    return pd.cut(values, bins=SEG_EDGES, labels=SEG_NAMES, right=False)

# ---- 1) Calibration widths trên 2023 (model fit ≤2022 — gap 1 năm, khôngContain 2023) ----
cal_df = df[df['year'] == 2023].copy()
fit22 = df[df['year'] <= 2022].copy()
make_te(fit22, [fit22, cal_df])
pre_cal = make_pre()
model21 = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
model21.fit(pre_cal.fit_transform(fit22[FEATURES]), fit22['resale_price'].values)
pred_cal = model21.predict(pre_cal.transform(cal_df[FEATURES]))
scores = np.abs(cal_df['resale_price'].values - pred_cal) / pred_cal
cal_buckets = bucket_of(pred_cal)

rel_w = {}
for i, g in enumerate(SEG_NAMES):
    msk = cal_buckets == g
    if msk.sum() >= 300:
        rel_w[g] = (float(np.quantile(scores[msk], ALPHA)), int(msk.sum()), 'group')
for i, g in enumerate(SEG_NAMES):
    if g not in rel_w:
        for j in range(1, len(SEG_NAMES)):
            for k in (i - j, i + j):
                if 0 <= k < len(SEG_NAMES) and SEG_NAMES[k] in rel_w:
                    rel_w[g] = (rel_w[SEG_NAMES[k]][0], rel_w[SEG_NAMES[k]][1], f'fallback←{SEG_NAMES[k]}')
                    break
            if g in rel_w:
                break
print("Relative widths (calibration trên 2023):")
for g in SEG_NAMES:
    w, n, src = rel_w[g]
    print(f"  {g:12s} n={n:5d}  w=±{w*100:5.1f}%  ({src})")

# ---- 2) Point model cuối: fit ≤2023, predict test 2024 (duy nhất một lần) ----
fit_df = df[df['year'] <= 2023].copy()
test_df = df[df['year'] == 2024].copy()
make_te(fit_df, [fit_df, test_df])
pre_final = make_pre()
X_fit = pre_final.fit_transform(fit_df[FEATURES])
y_fit = fit_df['resale_price'].values
X_test = pre_final.transform(test_df[FEATURES])
y_test = test_df['resale_price'].values

final_model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
final_model.fit(X_fit, y_fit)
pred = final_model.predict(X_test)

metrics = evaluate(y_test, pred)
metrics['n_test'] = len(y_test)
print("\n=== FINAL v3: test 2024 (mở duy nhất một lần) ===")
for k, v in metrics.items():
    print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ---- 3) Intervals trên test ----
w_test = np.array([rel_w[b][0] * p for b, p in zip(bucket_of(pred), pred)])
lo, hi = pred - w_test, pred + w_test
covered = (y_test >= lo) & (y_test <= hi)
print(f"\nCoverage khoảng 90% trên test 2024: {covered.mean()*100:.1f}%")

seg_rows = []
for seg, lo_p, hi_p in zip(SEG_NAMES, SEG_EDGES[:-1], SEG_EDGES[1:]):
    msk = (y_test >= lo_p) & (y_test < hi_p)
    if msk.sum() == 0:
        continue
    seg_rows.append({'Segment (giá thật)': seg, 'n': int(msk.sum()),
                     'Coverage_pct': float(covered[msk].mean() * 100),
                     'Mean_width': float(w_test[msk].mean()),
                     'MAE_q50_point': float(np.abs(y_test[msk] - pred[msk]).mean())})
seg_df = pd.DataFrame(seg_rows)
print(seg_df.to_string(index=False))

# ---- 4) Lưu bundle v3 ----
bundle = {
    'preprocessor': pre_final,
    'point_model': final_model,
    'street_te': {'stats': stats_of(fit_df), 'town_mean': fit_df.groupby('town')['resale_price'].mean(),
                  'prior': float(fit_df['resale_price'].mean()), 'm': M_SMOOTH},
    'seg_edges': SEG_EDGES, 'seg_names': SEG_NAMES, 'rel_widths': {k: v[0] for k, v in rel_w.items()},
    'alpha': ALPHA, 'metrics_test2024': metrics, 'coverage_test2024': float(covered.mean()),
}
joblib.dump(bundle, 'resale_price_model_v3.joblib')

out = test_df[['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm',
               'remaining_lease_years', 'resale_price']].copy()
out['predicted'] = pred
out['interval_lo'] = lo
out['interval_hi'] = hi
out['covered'] = covered
out.to_csv('v2_experiments/phase11_test2024_predictions_v3.csv', index=False)
pd.DataFrame([metrics]).to_csv('v2_experiments/phase11_final_metrics_v3.csv', index=False)
seg_df.to_csv('v2_experiments/phase11_conformal_segments_v3.csv', index=False)
print("\nĐã lưu: resale_price_model_v3.joblib + CSVs trong v2_experiments/")
