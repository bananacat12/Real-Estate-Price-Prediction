# HDB Resale Price Prediction — Time-Split Evaluation + Conformal Intervals

Dự đoán giá bán lại căn hộ HDB (Singapore) từ dữ liệu giao dịch **2015–2024 (~221,000 dòng)**.

Điểm đặc biệt của repo này không phải là model — mà là **quy trình đánh giá**: phát hiện và sửa
lỗi phương pháp của bản đầu tiên (random split trên dữ liệu thời gian), rồi xây bài toán lại từ đầu
với nguyên tắc *test trên năm chưa từng thấy* và *khoảng dự đoán có bảo chứng thống kê*.

## Kết quả trên test 2024 (năm hoàn toàn không xuất hiện trong quá trình train/tune)

| | v1 (random split — lỗi phương pháp) | v2 (time-split) | v3 (v2 + street TE + conformal) |
|---|---|---|---|
| R² | 0.9452 *(lạc quan ảo)* | 0.9008 | **0.9217** |
| RMSE | $39,955 | $58,478 | **$51,963** |
| MAE | $28,296 | $41,180 | **$37,175** |
| ±10% | 84.05% | 79.46% | **84.25%** |
| Đầu ra | 1 con số | 1 con số | Số + **khoảng tin cậy 90%** (coverage thực tế 92.9%) |
| Dung lượng model | ~250 MB | < 1 MB | < 1 MB |

> **Vì sao R² v1 "ảo"?** Dữ liệu giá không stationary (median tăng +33% từ 2015–2022 sang 2024).
> Random split cho model "nhìn trộm" phân phối giá của tương lai. Cross-validation 5-fold của chính
> v1 đã cho 0.881 — khoảng cách với 0.945 chính là mức lạc quan do split sai.

## Quy trình (và lý do từng quyết định)

```
Audit dữ liệu → Time-split → Baselines → Feature ablation → Target experiment
→ Tuning (year-based CV) → Final model → Conformal intervals → Error analysis
```

- **Time-based split**: train 2015–2022 (178,587) / validation 2023 (25,478) / test 2024 (16,906).
  Test chỉ được mở **đúng một lần** cho mỗi phiên bản model, ở bước cuối.
- **Mọi so sánh** (feature, transform, hyperparameter, model family) đều trên cùng validation 2023.
- **Feature ablation** thay vì tin feature importance: `storey_mid` giảm RMSE $4,290; mid-floor
  thắng one-hot; `distance_from_expressway` bị loại (vô dụng khi đã có storey — ΔRMSE ≈ 0).
- **Target encoding `street_name`** (566 giá trị) với 3 lớp chống leakage: fit chỉ trên train,
  smoothed `(n·mean + m·prior)/(n + m)` với m=10 (grid m ∈ {10, 50, 200}), street lạ fallback
  town-level. Giảm thêm $5,006 RMSE trên val.
- **Tuning thật sự**: LightGBM 20 trials + RandomForest 12 trials (có `max_features`), CV theo năm
  tường minh — (2015–2019→2020), (2015–2020→2021), (2015–2021→2022). LightGBM thắng RF trên cả
  CV lẫn val, đồng thời nhanh hơn ~300 lần.
- **Khoảng dự đoán bằng relative conformal calibration**: quantile regression thuần thất bại
  (coverage 53.5% thay vì 80%) nên thay bằng conformal — score `|residual|/prediction`, calibrate
  trên năm held-out (2023, model fit ≤2022), width theo nhóm giá dự đoán, nhóm thưa fallback
  nhóm gần nhất. Coverage thực tế trên test 2024: **92.9%** (mục tiêu 90%).

### Negative results (chạy thí nghiệm mới dám kết luận)

| Hypothesis | Kết quả |
|---|---|
| log1p(target) giảm heteroscedasticity | **Tệ hơn** trên mọi metric, cả LightGBM lẫn RF |
| Quantile regression (α=0.1/0.5/0.9) cho khoảng dự đoán | **Under-cover nghiêm trọng**: 53.5% coverage, segment >$1M chỉ 16.9% |

### Điểm yếu đã định lượng (không che giấu)

- Segment **>$1M**: chỉ 66.9% dự đoán rơi trong khoảng conformal 90% — năm calibration không có
  mẫu nào của nhóm này. Conformal đảm bảo coverage biên, không đảm bảo theo phân khúc.
- Point prediction: ±10% giảm theo giá — 86.8% ($300–500k) xuống ~60% ở town trung tâm đắt đỏ
  (Serangoon, Geylang, Kallang/Whampoa).

## Cấu trúc repo

```
ML_Cki_v2.ipynb        # notebook chính: 8 sections, chạy end-to-end ~2 phút
v2_experiments/        # provenance: script + CSV kết quả từng thí nghiệm (phase 0–11)
SUMMARY_v1_vs_v2.md    # so sánh 3 phiên bản + Q&A phỏng vấn
requirements.txt
```

## Chạy thử

```bash
pip install -r requirements.txt
jupyter notebook ML_Cki_v2.ipynb
```

Dùng model đã train (bundle gồm preprocessor + point model + conformal widths — file không commit,
train lại bằng notebook hoặc script `v2_experiments/phase11_final_v3.py`):

```python
import joblib, pandas as pd

bundle = joblib.load('resale_price_model_v3.joblib')
feat = pd.DataFrame([{
    'town': 'BUKIT BATOK', 'flat_type': '4 ROOM',
    'year': 2024, 'floor_area_sqm': 82.0, 'remaining_lease_years': 68,
    'storey_range': '07 TO 09', 'street_name': 'BUKIT BATOK STREET 32',
}])
sr = feat['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
feat['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

te = bundle['street_te']
name = feat['street_name'][0]
if name in te['stats'].index:
    st = te['stats'].loc[name]
    feat['street_te'] = (st['size'] * st['mean'] + te['m'] * te['prior']) / (st['size'] + te['m'])
else:
    feat['street_te'] = te['town_mean'].get(feat['town'][0], te['prior'])

cols = ['town', 'flat_type', 'year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid', 'street_te']
pred = bundle['point_model'].predict(bundle['preprocessor'].transform(feat[cols]))[0]
w = bundle['rel_widths']['$300–500k']
print(f"Giá dự đoán: ${pred:,.0f}")
print(f"Khoảng tin cậy 90%: ${pred - w * pred:,.0f} – ${pred + w * pred:,.0f}")
```

## Bước tiếp theo

1. Feature vị trí (khoảng cách MRT/CBD qua geocode) — hướng sửa chính cho segment cao và town trung tâm
2. `flat_model`, `month` từ dataset chính thức data.gov.sg (cho rolling price index + walk-forward theo quý)
3. Segment-specific calibration cho conformal ở nhóm >$1M

---
*Dữ liệu: giao dịch HDB resale Singapore 2015–2024. Toàn bộ số liệu trong README đều tái lập được
từ script trong `v2_experiments/`.*
