"""Phase 6 — Final model: refit LightGBM (params tốt nhất Phase 5) trên 2015-2023,
đánh giá DUY NHẤT MỘT LẦN trên test 2024. Không còn quyết định nào sau bước này.
Lưu: model joblib + metrics + dự đoán per-row cho Phase 7.
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
BEST_PARAMS = {'subsample_freq': 1, 'subsample': 1.0, 'num_leaves': 127, 'n_estimators': 400,
               'min_child_samples': 100, 'learning_rate': 0.05, 'colsample_bytree': 0.7}

def within_pct(y_true, y_pred, tol):
    lo, hi = y_true * (1 - tol), y_true * (1 + tol)
    return float(((y_pred >= lo) & (y_pred <= hi)).mean() * 100)

df = pd.read_csv('resale.csv')
sr = df['storey_range'].str.extract(r'(\d+)\s*TO\s*(\d+)')
df['storey_mid'] = (sr[0].astype(int) + sr[1].astype(int)) / 2

NUM = ['year', 'floor_area_sqm', 'remaining_lease_years', 'storey_mid']
CAT = ['town', 'flat_type']
FEATURES = NUM + CAT

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]).set_output(transform='pandas'), NUM),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]).set_output(transform='pandas'), CAT),
])

fit_df = df[df['year'] <= 2023]
test_df = df[df['year'] == 2024]
X_fit = pre.fit_transform(fit_df[FEATURES])
y_fit = fit_df['resale_price'].values
X_test = pre.transform(test_df[FEATURES])
y_test = test_df['resale_price'].values

model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1, **BEST_PARAMS)
model.fit(X_fit, y_fit)

pred = model.predict(X_test)
metrics = {
    'n_test': len(y_test),
    'R2': r2_score(y_test, pred),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
    'MAE': float(mean_absolute_error(y_test, pred)),
    'Pct_within_5pct': within_pct(y_test, pred, 0.05),
    'Pct_within_10pct': within_pct(y_test, pred, 0.10),
}
print("=== FINAL: LightGBM, train 2015-2023, test 2024 (chưa từng bị đụng đến) ===")
for k, v in metrics.items():
    print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Lưu artifacts
pre_full = Pipeline([('prep', pre), ('model', model)])
joblib.dump(pre_full, 'resale_price_model_v2.joblib')
out = test_df[['year', 'town', 'flat_type', 'storey_range', 'floor_area_sqm',
               'remaining_lease_years', 'resale_price']].copy()
out['predicted'] = pred
out.to_csv('v2_experiments/phase6_test2024_predictions.csv', index=False)
pd.DataFrame([metrics]).to_csv('v2_experiments/phase6_final_metrics.csv', index=False)
print("\nĐã lưu: resale_price_model_v2.joblib, phase6_test2024_predictions.csv, phase6_final_metrics.csv")
