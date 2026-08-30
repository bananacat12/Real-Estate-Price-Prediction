"""Phase 5 — Tuning với year-based CV, KHÔNG đụng validation 2023 và test 2024.
Folds (tường minh theo năm, trên tập train 2015-2022):
  Fold 1: train 2015-2019 -> val 2020
  Fold 2: train 2015-2020 -> val 2021
  Fold 3: train 2015-2021 -> val 2022
Scoring: R2 (không gian giá gốc). Sau CV, refit full train 2015-2022 và đánh giá MỘT LẦN trên val 2023
để chọn model family cho Phase 6.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RNG = 42
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

train = df[df['year'] <= 2022].copy()
val23 = df[df['year'] == 2023]
X_train = pre.fit_transform(train[FEATURES])
y_train = train['resale_price'].values
X_val23 = pre.transform(val23[FEATURES])
y_val23 = val23['resale_price'].values

years = train['year'].values
FOLDS = [(2019, 2020), (2020, 2021), (2021, 2022)]
idx = np.arange(len(train))

def cv_score(mk, params):
    scores = []
    for tr_max, va_year in FOLDS:
        tr_m = idx[years <= tr_max]
        va_m = idx[years == va_year]
        m = mk().set_params(**params)
        m.fit(X_train[tr_m], y_train[tr_m])
        scores.append(r2_score(y_train[va_m], m.predict(X_train[va_m])))
    return float(np.mean(scores)), float(np.std(scores))

grids = {
    'LightGBM': (
        lambda: lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1),
        {
            'n_estimators': [200, 400, 600, 1000],
            'learning_rate': [0.02, 0.05, 0.1],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 1.0],
            'subsample_freq': [1],
            'colsample_bytree': [0.7, 0.9, 1.0],
        },
        20,
    ),
    'RandomForest': (
        lambda: RandomForestRegressor(random_state=RNG, n_jobs=-1),
        {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 15, 25, 35],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 0.5],
        },
        12,
    ),
}

summary = []
for family, (mk, grid, n_iter) in grids.items():
    trials = list(ParameterSampler(grid, n_iter=n_iter, random_state=RNG))
    print(f"\n=== {family}: {n_iter} trials x 3 folds ===")
    best = None
    for i, params in enumerate(trials):
        t0 = time.time()
        mean_r2, std_r2 = cv_score(mk, params)
        print(f"  trial {i+1:02d}  CV-R2={mean_r2:.4f}±{std_r2:.4f}  {params}  ({time.time()-t0:.0f}s)")
        if best is None or mean_r2 > best[0]:
            best = (mean_r2, std_r2, params)
    # refit full train 2015-2022, đánh giá trên val 2023 (lần đầu dùng val sau khi chốt params)
    m = mk().set_params(**best[2])
    m.fit(X_train, y_train)
    from sklearn.metrics import mean_absolute_error
    pred23 = m.predict(X_val23)
    val_r2 = r2_score(y_val23, pred23)
    val_rmse = float(np.sqrt(mean_squared_error(y_val23, pred23)))
    val_mae = mean_absolute_error(y_val23, pred23)
    print(f"  BEST {family}: CV-R2={best[0]:.4f}±{best[1]:.4f} | val2023 R2={val_r2:.4f} RMSE={val_rmse:,.0f} MAE={val_mae:,.0f}")
    print(f"  BEST params: {best[2]}")
    summary.append({'Family': family, 'CV_R2_mean': best[0], 'CV_R2_std': best[1],
                    'Val2023_R2': val_r2, 'Val2023_RMSE': val_rmse, 'Val2023_MAE': val_mae,
                    'Params': str(best[2])})

pd.DataFrame(summary).to_csv('v2_experiments/phase5_tuning_summary.csv', index=False)
print("\nĐã lưu v2_experiments/phase5_tuning_summary.csv")
