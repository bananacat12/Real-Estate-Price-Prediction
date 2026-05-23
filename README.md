# 🏠 Singapore HDB Resale Price Prediction

> **Dự án Machine Learning** dự đoán giá bán lại căn hộ HDB tại Singapore  
> sử dụng **Random Forest** được tối ưu hóa bằng **RandomizedSearchCV**

`Python 3.8+` · `scikit-learn` · `XGBoost` · `LightGBM` · `MIT License`

---

## 📑 Mục Lục

- [Tổng Quan Dự Án](#-tổng-quan-dự-án)
- [Dataset](#-dataset)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Quy Trình Thực Hiện](#-quy-trình-thực-hiện)
- [Kết Quả](#-kết-quả)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)

---

## 🎯 Tổng Quan Dự Án

Thị trường bất động sản Singapore, đặc biệt phân khúc căn hộ HDB (Housing & Development Board), biến động theo nhiều yếu tố kinh tế – xã hội. Dự án này xây dựng mô hình Machine Learning có khả năng **dự đoán giá bán lại (resale price)** dựa trên vị trí, diện tích, tầng, loại căn hộ và thời hạn thuê còn lại.

### Mục tiêu chính

- Phân tích và khám phá dữ liệu (EDA) toàn diện về thị trường HDB Singapore
- So sánh hiệu năng 4 mô hình: Linear Regression, Random Forest, XGBoost, LightGBM
- Tối ưu hóa siêu tham số với RandomizedSearchCV
- Đánh giá mô hình đa chiều: metrics, residual analysis, accuracy theo từng khu vực

---

## 📊 Dataset

| Thuộc tính | Thông tin |
|---|---|
| **Số bản ghi** | 220,971 giao dịch |
| **Số cột** | 11 đặc trưng |
| **Khoảng thời gian** | 2015 – 2024 |
| **Missing Values** | Không có (0 trên tất cả cột) |
| **File** | `resale.csv` |

### Mô Tả Các Cột

| Cột | Kiểu | Mô Tả |
|---|---|---|
| `year` | int | Năm giao dịch (2015–2024) |
| `town` | str | Khu vực / quận (26 khu vực) |
| `flat_type` | str | Loại căn hộ (1 ROOM → MULTI-GENERATION) |
| `block` | str | Số block tòa nhà |
| `street_name` | str | Tên đường |
| `storey_range` | str | Khoảng tầng thô (ví dụ: "07 TO 09") |
| `floor_area_sqm` | float | Diện tích sàn (m²) |
| `remaining_lease_years` | int | Số năm hợp đồng thuê còn lại |
| `resale_price` | float | **🎯 Biến mục tiêu** – Giá bán lại (SGD) |
| `storey_range_category` | str | Phân nhóm tầng: Low / Low-Mid / Mid / High / Very High |
| `distance_from_expressway` | str | Khoảng cách đến đường cao tốc gần nhất |

### Thống Kê Biến Mục Tiêu (`resale_price`)

| Chỉ số | Giá trị (SGD) |
|---|---|
| Min | 140,000 |
| 25th Percentile | 365,000 |
| Median | 458,000 |
| Mean | ~490,867 |
| 75th Percentile | 585,000 |
| Max | 1,588,000 |

> 💡 Phân phối giá lệch phải (right-skewed) — đa số giao dịch nằm trong khoảng **300k–500k SGD**. Phát hiện **4,902 ngoại lai** ở `resale_price` và **1,268 ngoại lai** ở `floor_area_sqm` — tất cả được **giữ lại** vì phản ánh thực tế căn hộ cao cấp.

### Ma Trận Tương Quan (với `resale_price`)

| Đặc trưng | Tương quan |
|---|---|
| `floor_area_sqm` | **0.595** ← mạnh nhất |
| `remaining_lease_years` | 0.321 |
| `year` | 0.316 |

---

## 📁 Cấu Trúc Dự Án

```
📦 hdb-resale-price-prediction/
├── 📓 Final.ipynb                               # Notebook chính (EDA + Training + Evaluation)
├── 📄 resale.csv                                # Dataset gốc
│
├── 🤖 random_forest_resale_price_model.joblib   # Mô hình đã huấn luyện
│
├── 📊 model_evaluation_metrics.csv              # Bảng MSE, RMSE, MAE, R²
├── 📊 accuracy_by_town.csv                      # Độ chính xác ±10% theo khu vực
├── 📊 cross_validation_results.csv              # Kết quả 5-Fold Cross-Validation
│
├── 🖼️ residuals_distribution.png
├── 🖼️ residuals_vs_predicted.png
├── 🖼️ actual_vs_predicted_with_line.png
├── 🖼️ feature_importances.png
│
└── 📄 README.md
```

---

## ⚙️ Cài Đặt

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib scipy
```

```bash
jupyter notebook Final.ipynb
```

---

## 🔄 Quy Trình Thực Hiện

### 1. 🔍 Phân Tích Khám Phá Dữ Liệu (EDA)

| Bước | Nội dung | Công cụ |
|---|---|---|
| Tổng quan | `data.info()`, `data.describe()` | pandas |
| Missing Values | Heatmap — **không có giá trị thiếu** | seaborn |
| Phân phối giá | Histogram → lệch phải | matplotlib |
| Tương quan | Heatmap & Pairplot | seaborn |
| Biến phân loại | Countplot cho 4 biến categorical | seaborn |
| Ngoại lai | Boxplot + IQR tùy chỉnh | seaborn |
| Feature mới | `price_per_sqm` = `resale_price / floor_area_sqm` | pandas |
| Xu hướng thời gian | Boxplot + Line Chart theo năm | seaborn |

> **Quyết định:** Giữ lại ngoại lai → ưu tiên tree-based model thay vì Linear Regression.

---

### 2. 🛠️ Tiền Xử Lý Dữ Liệu

**Xử lý `distance_from_expressway`** (chuỗi đặc biệt → số):

| Định dạng gốc | Kết quả | Ví dụ |
|---|---|---|
| `"101-150m"` | Trung bình khoảng | → 125.0 |
| `"<=50m"` | Giá trị biên | → 50.0 |
| `">500m"` | Giá trị biên | → 500.0 |

**Phân chia dữ liệu:** 80% Train (~176,776) / 20% Test (~44,195)

**Pipeline tiền xử lý:**

| Loại | Cột | Xử lý |
|---|---|---|
| Numerical | `year`, `floor_area_sqm`, `remaining_lease_years`, `distance_from_expressway` | Median Imputer → StandardScaler |
| Categorical | `town`, `flat_type`, `storey_range_category` | Mode Imputer → OneHotEncoder |
| High Cardinality | `block`, `street_name` | Mode Imputer → **TargetEncoder** |

---

### 3. 🏁 So Sánh Baseline Models

| Mô hình | RMSE | MAE | R² | Training Time |
|---|---|---|---|---|
| **Random Forest** | **32,836** | **22,998** | **0.9630** | 600.64s |
| XGBoost | 35,136 | 25,606 | 0.9577 | 0.92s |
| LightGBM | 40,495 | 29,392 | 0.9438 | 0.76s |
| Linear Regression | 84,975 | 63,420 | 0.7523 | 0.38s |

> Random Forest đạt R² cao nhất (0.963) → được chọn làm mô hình chính để tinh chỉnh.

---

### 4. 🎛️ Tối Ưu Hóa Siêu Tham Số

**Cấu hình RandomizedSearchCV:** 20 tổ hợp, 3-fold CV, scoring = R²

**Siêu tham số tốt nhất tìm được:**

| Tham số | Giá trị |
|---|---|
| `n_estimators` | 72 |
| `max_depth` | 30 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |

---

## 📈 Kết Quả

### Hiệu Năng Mô Hình Cuối (Random Forest Sau Tuning)

| Chỉ số | Train | Test |
|---|---|---|
| **RMSE** | 25,196.50 | **33,327.11** |
| **MAE** | 17,968.03 | **23,177.07** |
| **R²** | 0.9781 | **0.9619** |
| MSE | 634,863,401 | 1,110,696,013 |

> ✅ Chênh lệch R² Train–Test = 0.016 < 0.1 → **Không có dấu hiệu overfitting**

### Độ Chính Xác Trong Khoảng Sai Số

| Khoảng sai số | Tỷ lệ dự đoán đạt |
|---|---|
| **±10%** | **89.53%** |
| **±5%** | 63.59% |

### Cross-Validation (5-Fold trên tập Train)

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Trung bình** |
|---|---|---|---|---|---|
| 0.9596 | 0.9585 | 0.9605 | 0.9597 | 0.9592 | **0.9595** |

> ✅ Các fold rất đồng đều — mô hình ổn định, không phụ thuộc vào phân chia dữ liệu.

### Độ Chính Xác ±10% Theo Từng Khu Vực

| Hạng | Khu vực | Accuracy ±10% |
|---|---|---|
| 🥇 1 | PUNGGOL | 94.81% |
| 🥈 2 | SENGKANG | 93.16% |
| 🥉 3 | SEMBAWANG | 93.16% |
| 4 | WOODLANDS | 92.49% |
| 5 | TAMPINES | 91.84% |
| ... | ... | ... |
| 22 | BUKIT TIMAH | 84.75% |
| 23 | JURONG EAST | 84.26% |
| 24 | MARINE PARADE | 83.33% |
| 25 | BISHAN | 82.72% |
| 26 | TOA PAYOH | 82.17% |

> Mô hình dự đoán tốt nhất ở các khu vực mới phát triển đồng đều (Punggol, Sengkang), kém hơn ở các khu vực trung tâm có giá biến động phức tạp (Toa Payoh, Bishan).

---

### Tải và Sử Dụng Mô Hình

```python
import joblib
import pandas as pd
import numpy as np

# Load mô hình
model = joblib.load('random_forest_resale_price_model.joblib')

# Hàm tiền xử lý distance_from_expressway
def preprocess_distance(df):
    def convert(val):
        if isinstance(val, str):
            val = val.replace('m', '').strip()
            if '-' in val:
                lo, hi = val.split('-')
                return (float(lo) + float(hi)) / 2
            elif val.startswith('<='):
                return float(val.replace('<=', ''))
            elif val.startswith('>'):
                return float(val.replace('>', ''))
            else:
                return float(val)
        return np.nan
    df['distance_from_expressway'] = df['distance_from_expressway'].apply(convert)
    return df

# Chuẩn bị dữ liệu mới
new_data = pd.DataFrame([{
    'year': 2024,
    'town': 'BISHAN',
    'flat_type': '4 ROOM',
    'floor_area_sqm': 93,
    'remaining_lease_years': 75,
    'distance_from_expressway': '151-300m',
    'storey_range_category': 'Mid (13-18)',
    'block': '123',
    'street_name': 'BISHAN ST 22'
}])

new_data = preprocess_distance(new_data)
predicted_price = model.predict(new_data)
print(f"Giá dự đoán: SGD {predicted_price[0]:,.0f}")
```

---

## 🧰 Công Nghệ Sử Dụng

| Thư viện | Mục đích |
|---|---|
| `pandas`, `numpy` | Xử lý và phân tích dữ liệu |
| `matplotlib`, `seaborn` | Trực quan hóa |
| `scikit-learn` | Pipeline, preprocessing, Random Forest, metrics, cross-validation |
| `xgboost` | Mô hình XGBoost (so sánh baseline) |
| `lightgbm` | Mô hình LightGBM (so sánh baseline) |
| `scipy` | `randint` cho RandomizedSearchCV |
| `joblib` | Lưu và load mô hình |

---

## 📌 Các Quyết Định Kỹ Thuật Quan Trọng

| Vấn đề | Quyết định | Lý do |
|---|---|---|
| Ngoại lai giá cao (4,902 điểm) | **Giữ lại** | Phản ánh thực tế căn hộ cao cấp |
| Transform target | **Không log** | Tree-based models không nhạy với phân phối lệch |
| Encode `block`, `street_name` | **TargetEncoder** | Số lượng unique lớn, tránh bùng nổ chiều |
| Chọn mô hình chính | **Random Forest** | R² cao nhất (0.963) trong baseline |
| Tìm siêu tham số | **RandomizedSearchCV** | Hiệu quả hơn GridSearch với không gian lớn |

---

*🏠 Singapore HDB Resale Price Prediction — Machine Learning Pipeline with Random Forest*
