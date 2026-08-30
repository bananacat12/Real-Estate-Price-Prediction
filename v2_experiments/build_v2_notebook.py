"""Build ML_Cki_v2.ipynb — notebook portfolio sạch: time-split, LightGBM, evaluation 2024."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Dự đoán giá HDB Resale Singapore — v2 (evaluation sạch)

**Khác biệt cốt lõi so với v1:** thay random split bằng **time-based split**; test 2024 được giữ
*untouched* cho đến cell đánh giá cuối cùng. Mọi quyết định (feature, transform, hyperparameter,
model family) đều được chọn trên **validation 2023** hoặc year-based CV trong tập train.

| Tập | Năm | Số mẫu |
|---|---|---|
| Train | 2015–2022 | 178,587 |
| Validation | 2023 | 25,478 |
| Test | 2024 | 16,906 *(không đụng đến trước cell Final)* |

**Kết quả cuối (test 2024, LightGBM):** R² **0.9008** · RMSE **$58,478** · MAE **$41,180** ·
46.99% dự đoán trong ±5% · **79.46% trong ±10%**.

Các thí nghiệm ablation / target-transform / tuning được chạy bằng script trong `v2_experiments/`
(kết quả load từ CSV ở đây để notebook chạy nhanh; script có trong repo nên tái lập được 100%).""")

md("## 1. Load dữ liệu & audit")
code("""import numpy as np
import pandas as pd

df = pd.read_csv('resale.csv')
print(f"Số dòng: {len(df):,} | Cột: {list(df.columns)}")
print(f"Trùng lặp hoàn toàn: {df.duplicated().sum()} | Thiếu giá trị: {df.isnull().sum().sum()}")

by_year = df.groupby('year')['resale_price'].agg(['size', 'median'])
by_year.columns = ['n_giao_dịch', 'median_giá']
by_year"""
)
md("""**Audit — kết luận chính:**
- Median giá tăng $400–410k (2015–2019) → $580k (2024): dữ liệu **không stationary**. Random split
  sẽ cho mô hình nhìn thấy phân phối giá của tương lai → chỉ số đánh giá lạc quan ảo. Đó là lỗi
  phương pháp chính của v1.
- Không có null, không trùng dòng. Tất cả feature đều biết được tại thời điểm niêm yết (point-in-time).
- `storey_range` (17 giá trị, format `XX TO YY`) có tín hiệu giá đơn điệu mạnh: $413k (tầng 1–3)
  → $1.11M (tầng 49–51). `distance_from_expressway` tín hiệu yếu, 86% là ">500m".""")

md("## 2. Feature engineering & time-based split")
code("""# distance_from_expressway: 6 bậc có thứ tự -> numeric theo midpoint
DIST_MAP = {'<=50m': 25, '51-100m': 75, '101-150m': 125, '151-300m': 225, '301-500m': 400, '>500m': 500}
df['distance_ord'] = df['distance_from_expressway'].map(DIST_MAP)

# storey_range '04 TO 06' -> điểm giữa tầng 5.0
sr = df['storey_range'].str.extract(r'(\\d+)\\s*TO\\s*(\\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = NUM + CAT

train = df[df['year'] <= 2022]   # tuning + fitting
val   = df[df['year'] == 2023]   # mọi so sánh model/feature
test  = df[df['year'] == 2024]   # KHÔNG đụng đến trước cell cuối
print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")"""
)
md("""**Feature selection có bằng chứng (ablation, LightGBM, đánh giá trên val 2023):**

| Cấu hình | R² | RMSE | ±10% |
|---|---|---|---|
| base (5 features, không storey/distance) | 0.8501 | $67,357 | 70.4% |
| base + distance | 0.8520 | $66,933 | 70.6% |
| base + storey (one-hot) | 0.8653 | $63,836 | 73.6% |
| **base + storey_mid** | **0.8686** | **$63,067** | **73.7%** |
| base + dist + storey_mid | 0.8671 | $63,418 | 73.8% |

→ `storey_mid` giảm RMSE $4,290; mid-floor thắng one-hot; **distance bị loại** (vô dụng khi đã có
storey — trả lời definitively câu hỏi v1 bỏ ngỏ). Negative result khác: **log1p(target) tệ hơn raw**
trên mọi metric với cả LightGBM lẫn RF → giữ raw target (chi tiết: `v2_experiments/phase4_target.csv`).""")

md("## 3. Baselines (đánh giá trên val 2023)")
code("""import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def within_pct(y_true, y_pred, tol):
    return float(((y_pred >= y_true * (1 - tol)) & (y_pred <= y_true * (1 + tol))).mean() * 100)

def evaluate(y_true, y_pred):
    return {'R2': r2_score(y_true, y_pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'Pct_within_10pct': within_pct(y_true, y_pred, 0.10)}

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]).set_output(transform='pandas'), CAT),
])

X_val = pre.fit(train[FEATURES]).transform(val[FEATURES])
y_val = val['resale_price'].values
X_train_all = pre.transform(train[FEATURES])
y_train = train['resale_price'].values

baseline_rows = []
for name, model in [
    ('Median', DummyRegressor(strategy='median')),
    ('Ridge', Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=1.0))])),
    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('LightGBM', lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
]:
    model.fit(X_train_all, y_train)
    r = evaluate(y_val, model.predict(X_val)); r['Model'] = name
    baseline_rows.append(r)
    print(f"{name:14s} R2={r['R2']:.4f} RMSE={r['RMSE']:>10,.0f} MAE={r['MAE']:>9,.0f} ±10%={r['Pct_within_10pct']:5.1f}%")"""
)
md("""Baseline Median cho R² âm (−0.62) vì median train ($435k) lệch hẳn median val 2023 ($550k) —
minh chứng trực tiếp cho mức dịch phân phối theo thời gian. LightGBM default ngang RF nhưng nhanh
hơn ~300 lần → dùng LightGBM cho tuning.""")
code("""pd.DataFrame(baseline_rows)[['Model','R2','RMSE','MAE','Pct_within_10pct']]"""
)

md("""## 4. Tuning — year-based CV, không đụng val/test
Folds: (2015–2019→2020), (2015–2020→2021), (2015–2021→2022). LightGBM 20 trials, RF 12 trials
(có `max_features`). Kết quả đầy đủ: `v2_experiments/phase5_tuning_summary.csv`.""")
code("""tuning = pd.read_csv('v2_experiments/phase5_tuning_summary.csv')
tuning[['Family', 'CV_R2_mean', 'CV_R2_std', 'Val2023_R2', 'Val2023_RMSE', 'Val2023_MAE']]"""
)
md("""LightGBM thắng cả CV (0.8556 vs 0.8503) lẫn val 2023 (0.8888 vs 0.8831) → chốt LightGBM với
params: `n_estimators=400, learning_rate=0.05, num_leaves=127, min_child_samples=100,
colsample_bytree=0.7, subsample=1.0`. RF vẫn được giữ làm sanity-check ở Phase 2/4.""")

md("## 5. FINAL — train 2015–2023, đánh giá duy nhất một lần trên test 2024")
code("""BEST_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=127,
                   min_child_samples=100, colsample_bytree=0.7, subsample=1.0, subsample_freq=1)

fit_df = df[df['year'] <= 2023]
X_fit = pre.fit_transform(fit_df[FEATURES])
y_fit = fit_df['resale_price'].values
X_test = pre.transform(test[FEATURES])
y_test = test['resale_price'].values

final_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, **BEST_PARAMS)
final_model.fit(X_fit, y_fit)
y_pred = final_model.predict(X_test)

final_metrics = evaluate(y_test, y_pred)
print("=== Test 2024 (chưa từng bị đụng đến) ===")
for k, v in final_metrics.items():
    print(f"  {k}: {v:,.4f}")"""
)
md("""**So sánh với v1:** v1 báo test R² 0.9452 / ±10% 84.05% — nhưng từ random split, nơi tập test
chứa đúng phân phối giá của các năm đã học. Con số này của v2 (R² 0.9008 trên **năm hoàn toàn
chưa thấy**) mới là ước lượng đáng tin cho câu hỏi "mô hình dự đoán năm tới tốt đến đâu".
R² giảm không phải vì model yếu hơn — LightGBM ở đây mạnh hơn RF của v1 — mà vì cách đo trung thực hơn.""")

md("## 6. Phân tích lỗi")
code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

preds = test[['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm',
              'remaining_lease_years', 'resale_price']].copy()
preds['predicted'] = y_pred
resid = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].scatter(y_test, y_pred, alpha=0.3, s=6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set(xlabel='Actual (SGD)', ylabel='Predicted (SGD)', title='Actual vs Predicted — test 2024')
axes[1].scatter(y_pred, resid, alpha=0.3, s=6)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set(xlabel='Predicted (SGD)', ylabel='Residual (SGD)', title='Residuals vs Predicted — test 2024')
plt.tight_layout(); plt.savefig('v2_analysis_charts.png', dpi=110); plt.show()"""
)
code("""segments = [(0, 300_000, '<$300k'), (300_000, 500_000, '$300–500k'), (500_000, 700_000, '$500–700k'),
            (700_000, 1_000_000, '$700k–1M'), (1_000_000, np.inf, '>$1M')]
rows = []
for lo, hi, name in segments:
    m = (y_test >= lo) & (y_test < hi)
    rows.append({'Segment': name, 'n': int(m.sum()),
                 'MAE': float(np.abs(resid[m]).mean()),
                 'Pct_within_10pct': within_pct(y_test[m], y_pred[m], 0.10)})
seg = pd.DataFrame(rows)
seg"""
)
md("""**Phát hiện then chốt:** MAE tăng gần tuyến tính theo phân khúc giá và phân khúc >$1M chỉ có
34.1% dự đoán trong ±10% (so với 86.8% ở $300–500k). RMSE trung bình của toàn tập đang che giấu
việc model kém nhất ở đúng nơi giá trị tiền lớn nhất. Hướng cải thiện tiếp theo: feature khu vi mô
(đã có `street_name` — thử target encoding), hoặc model riêng cho phân khúc cao / quantile loss.
Log-transform **không** phải giải pháp — đã thử và tệ hơn (Phase 4).""")
code("""# Permutation importance ở mức feature (ΔRMSE khi xáo trộn; test 2024, 5 repeats)
rng = np.random.default_rng(42)
test_feat = test[FEATURES].reset_index(drop=True)
base_rmse = final_metrics['RMSE']
imp = []
for feat in FEATURES:
    deltas = []
    for _ in range(5):
        sh = test_feat.copy()
        sh[feat] = rng.permutation(sh[feat].values)
        p = final_model.predict(pre.transform(sh))
        deltas.append(np.sqrt(((y_test - p) ** 2).mean()) - base_rmse)
    imp.append({'Feature': feat, 'dRMSE': float(np.mean(deltas)), 'std': float(np.std(deltas))})
pd.DataFrame(imp).sort_values('dRMSE', ascending=False)"""
)
md("""Lưu ý: `year` có ΔRMSE = 0 — vì test 2024 chỉ chứa đúng một giá trị năm, xáo trộn không đổi gì.
Vai trò của `year` (proxy xu hướng giá) nằm trong quá trình học, không đo được trên test 1 năm.""")

md("""## 7. Tổng kết cho phỏng vấn
1. **Vấn đề của v1:** random split trên dữ liệu thời gian (không stationary, median +33% từ train
   sang test) → R² 0.945 lạc quan. Tuning với search space hẹp bao quanh giá trị đã biết.
2. **Sửa:** time-based split (train ≤2022, val 2023, test 2024 untouched); year-based CV cho tuning;
   mọi so sánh trên cùng một validation.
3. **Giá trị tạo ra:** ablation chứng minh `storey` đáng $4.3k RMSE và `distance` vô dụng;
   negative result về log-transform; LightGBM thay RF (nhanh ~300×, chính xác hơn).
4. **Kết quả trung thực:** R² 0.9008, ±10% 79.5% trên năm chưa thấy — và biết rõ model yếu ở đâu
   (phân khúc >$1M: 34.1% ±10%; các town trung tâm: ~58–63%).
5. **Bước tiếp theo:** target encoding `street_name`, quantile loss / model riêng cho segment cao.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12.6'},
}
with open('ML_Cki_v2.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Đã tạo ML_Cki_v2.ipynb với", len(cells), "cells")
