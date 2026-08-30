"""Append section 8 (v3 upgrades: street TE + conformal intervals) vào ML_Cki_v2.ipynb."""
import nbformat as nbf

nb = nbf.read('ML_Cki_v2.ipynb', as_version=4)

def md(s): nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s): nb.cells.append(nbf.v4.new_code_cell(s))

md("""## 8. Nâng cấp v3: street target-encoding + khoảng dự đoán conformal

Hai nâng cấp khả thi hoàn toàn với dataset hiện có, giữ nguyên protocol time-split:
quyết định chốt trên **val 2023**, test 2024 chỉ mở ở cuối (chi tiết script: `v2_experiments/phase9–11`).

### 8.1. Target encoding `street_name` (566 giá trị) — 3 lớp chống leakage
Fit encoding **chỉ trên train**, smoothing `(n·mean + m·prior)/(n + m)` với `m=10` (thí nghiệm m∈{10,50,200}),
street lạ → fallback town-level (val 2023 chỉ 12/25,478 dòng rơi vào trường hợp này).""")
code("""te_results = pd.read_csv('v2_experiments/phase9_street_te.csv')
te_results""")
md("""**Kết quả (val 2023):** raw-price TE m=10 giảm RMSE **$58,004 → $52,998 (−$5,006)**, R² 0.8888 → 0.9072.
Biến thể "premium index" chuẩn hóa point-in-time cho kết quả tương đương (0.9063–0.9068) → chọn
biến thể đơn giản hơn. So với Phase 3, đây là feature đơn lẻ mạnh nhất kể từ `storey_mid`.

### 8.2. Khoảng dự đoán: quantile thuần thất bại → conformal phân đoạn
**Negative result đáng nhớ:** LGBM quantile (α=0.1/0.5/0.9) chỉ phủ **53.5%** (kỳ vọng 80%), segment
>$1M chỉ 16.9% — một model quantile toàn cục không nắm được heteroscedasticity theo phân khúc.

Giải pháp: **relative conformal calibration** — score `|resid|/pred` calibrate trên năm 2023
(model fit ≤2022, gap 1 năm), width theo nhóm giá dự đoán, nhóm thưa fallback nhóm gần nhất.""")
code("""q_seg = pd.read_csv('v2_experiments/phase10_quantile_segments.csv')
print('Quantile thuần (val 2023) — coverage mục tiêu ~80%:')
print(q_seg.to_string(index=False))
v3_seg = pd.read_csv('v2_experiments/phase11_conformal_segments_v3.csv')
print('\\nConformal relative (test 2024) — coverage mục tiêu 90%:')
print(v3_seg.to_string(index=False))""")
md("""### 8.3. FINAL v3 — test 2024 mở duy nhất một lần""")
code("""v3_metrics = pd.read_csv('v2_experiments/phase11_final_metrics_v3.csv')
print('=== FINAL v3 (LightGBM + street TE, train 2015–2023) — test 2024 ===')
print(v3_metrics.T.rename(columns={0: 'value'}))
compare = pd.DataFrame({
    'Phiên bản': ['v1 (random split — lạc quan)', 'v2 (time-split)', 'v3 (time-split + street TE)'],
    'R2': [0.9452, 0.9008, 0.9217],
    'RMSE': [39955, 58478, 51963],
    'MAE': [28296, 41180, 37175],
    'Pct_within_10pct': [84.05, 79.46, 84.25],
})
compare""")
md("""**Đọc kết quả:** v3 đạt **R² 0.9217 / ±10% 84.25%** trên năm chưa từng thấy — gần như san bằng
con số 0.945 "ảo" của v1 nhưng với phương pháp sạch. Coverage conformal toàn tập **92.9%**
(mục tiêu 90%).

**Hạn chế báo cáo trung thực:** segment >$1M chỉ được phủ 66.9% — nhóm này có 0 mẫu >$1M trong
năm calibration 2023 (fallback width từ nhóm $700k–1M). Conformal đảm bảo coverage *biên* (marginal);
coverage *theo phân khúc* cần thêm dữ liệu calibration cho nhóm cao. Đây là limitation chính thức
của v3 và là bước tiếp theo rõ ràng nhất.""")

with open('ML_Cki_v2.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Đã append", len(nb.cells), "cells tổng cộng")
