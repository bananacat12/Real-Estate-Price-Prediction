# Tổng kết v1 → v2 → v3 + talking points phỏng vấn

## Bảng so sánh v1 vs v2 vs v3

| | v1 (`ML_Cki.ipynb`) | v2 (`ML_Cki_v2.ipynb`) | v3 (nâng cấp trong notebook) |
|---|---|---|---|
| Split | Random 80/20 (test lẫn lộn 2015–2024) | Time-based: train 2015–2022 / val 2023 / **test 2024 untouched** | Giữ nguyên v2 |
| Model | RandomForest, max_depth=None | LightGBM (tuned, year-based CV) | LightGBM + **street target-encoding** |
| Features | year, town, flat_type, floor_area, lease, distance | + **storey_mid**; bỏ distance (ablation) | + **street_te** (smoothed m=10) |
| Tuning | Search space hẹp bao quanh giá trị đã biết | 20 trials LGBM + 12 trials RF, CV theo năm | Giữ nguyên |
| Đầu ra | 1 con số dự đoán | 1 con số dự đoán | Con số + **khoảng tin cậy 90% (conformal)** |
| **Test R²** | 0.9452 *(lạc quan)* | 0.9008 | **0.9217** |
| Test RMSE | $39,955 | $58,478 | **$51,963** |
| Test MAE | $28,296 | $41,180 | **$37,175** |
| ±10% | 84.05% | 79.46% | **84.25%** |
| Coverage khoảng 90% | — | — | **92.9%** toàn tập; >$1M: 66.9% *(limitation)* |
| Model size | ~250 MB | < 1 MB | < 1 MB |


**Đọc đúng con số:** R² giảm không phải vì model yếu hơn — LightGBM của v2 thắng RF của v1 trên
cùng validation (0.8888 vs 0.8831). R² giảm vì **cách đo** trung thực hơn: v1 cho model nhìn thấy
phân phối giá của tương lai (median giá +33% từ train sang test), v2 đo trên năm hoàn toàn mới.

## Kết quả thí nghiệm đáng nhớ

1. **Ablation** (val 2023): storey_mid giảm RMSE $4,290 (67.4k→63.1k); mid-floor thắng one-hot;
   distance vô dụng khi đã có storey (thậm chí hơi tệ hơn: 0.8686 vs 0.8671).
2. **Negative result — log1p target tệ hơn raw** trên mọi metric với cả LightGBM lẫn RF. Hetero-
   scedasticity có thật nhưng log không phải giải pháp cho tree model.
3. **Negative result — quantile regression thuần under-cover**: coverage 53.5% (mục tiêu 80%),
   segment >$1M chỉ 16.9%. Một model quantile toàn cục không nắm được heteroscedasticity.
4. **Street target-encoding** (fit chỉ trên train, smoothed m=10, fallback town-level): RMSE val
   giảm thêm $5,006 (58.0k→53.0k), R² 0.8888→0.9072. Feature đơn lẻ mạnh nhất kể từ storey_mid.
5. **Relative conformal calibration** (score |resid|/pred, calibrate trên năm held-out, width theo
   nhóm giá dự đoán): coverage test 2024 = 92.9% (mục tiêu 90%). Limitation: segment >$1M chỉ 66.9%
   vì nhóm này không có mẫu trong năm calibration — conformal đảm bảo coverage biên, không đảm bảo
   theo phân khúc.
6. **Price segment** (test 2024, v2): ±10% từ 86.8% ($300–500k) xuống 34.1% (>$1M); v3 (với street
   TE) cải thiện điểm yếu này ở mức point-prediction.
7. **Baseline Median R² = −0.62** trên val 2023: bằng chứng trực tiếp cho mức dịch phân phối.

## Talking points khi bị hỏi tra xoáy

- *"Tại sao không tin R² 0.945 của bản cũ?"* — Random split trên dữ liệu không stationary; CV 5-fold
  của chính bản cũ đã cho 0.881, và val 2023 của tôi cho ~0.87–0.89. Con số 0.945 chỉ tồn tại khi
  test chứa phân phối giá của các năm đã học.
- *"Tại sao LightGBM?"* — Ngang RF về accuracy, nhanh ~300 lần (0.9s vs 305s), model <1MB vs 250MB.
- *"Model yếu ở đâu?"* — >$1M và các town trung tâm. Đã cải thiện bằng street target-encoding
  (R² 0.9008→0.9217); phần còn lại cần dữ liệu vị trí (MRT/CBD) ngoài dataset này.
- *"Log target có giúp không?"* — Không, đã thử: tệ hơn trên mọi metric với cả 2 model family.
- *"Quantile regression cho khoảng dự đoán được không?"* — Thuần quantile thì không: under-cover
  nghiêm trọng (53.5%). Conformal calibration mới là câu trả lời đúng (92.9% coverage thực tế).
- *"Khoảng dự đoán tin được đến đâu?"* — Coverage toàn tập 92.9% với mức mục tiêu 90%, nhưng bị
  over-cover ở segment thấp (95%) và under-cover ở >$1M (66.9%) — tôi biết chính xác giới hạn này
  và nguyên nhân của nó (thiếu mẫu calibration ở segment cao).
- *"Feature nào quyết định?"* — floor_area (ΔRMSE $110k), town ($79k), lease ($66k) theo permutation
  importance mức feature trên test 2024.

## Artifacts

- `ML_Cki_v2.ipynb` — notebook chính (8 sections gồm cả nâng cấp v3), chạy end-to-end ~2 phút, 0 cell lỗi
- `resale_price_model_v2.joblib` — pipeline v2 (tham chiếu)
- `resale_price_model_v3.joblib` — bundle cuối: preprocessor + point model + conformal widths
- `v2_experiments/` — toàn bộ script + CSV kết quả từng phase 0–11 (provenance đầy đủ)
- `requirements.txt`
