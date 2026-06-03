"""
模型训练与预测模块

优化要点:
- StandardScaler: 线性模型自动标准化
- TimeSeriesSplit CV: 时序交叉验证
- Log 变换: 对右偏的单价做 log1p 变换，残差更接近正态分布
- XGBoost: 通常比 GradientBoosting 精度更高
- GridSearchCV: 自动搜索最优超参数
- 集成预测: 多个模型加权平均
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 尝试导入 XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ========== 模型配置 ==========
MODEL_CONFIGS = {
    "LinearRegression": {
        "class": LinearRegression,
        "label": "线性回归",
        "params": {},
        "need_scaler": True,
        "description": "基准线性模型",
    },
    "Ridge": {
        "class": Ridge,
        "label": "岭回归",
        "params": {"alpha": 10.0},
        "need_scaler": True,
        "description": "L2 正则化，防过拟合",
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "label": "随机森林",
        "params": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5,
                   "random_state": 42, "n_jobs": -1},
        "need_scaler": False,
        "description": "集成决策树，非线性",
    },
    "GradientBoosting": {
        "class": GradientBoostingRegressor,
        "label": "梯度提升",
        "params": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5,
                   "learning_rate": 0.05, "subsample": 0.8, "random_state": 42},
        "need_scaler": False,
        "description": "逐步优化残差",
    },
}

# XGBoost 如果可用
if HAS_XGBOOST:
    MODEL_CONFIGS["XGBoost"] = {
        "class": XGBRegressor,
        "label": "XGBoost",
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                   "subsample": 0.8, "colsample_bytree": 0.8,
                   "random_state": 42, "n_jobs": -1},
        "need_scaler": False,
        "description": "梯度提升最优实现",
    }

# 简化的超参数网格（用于 GridSearchCV）
PARAM_GRIDS = {
    "Ridge": {"estimator__alpha": [0.1, 1.0, 10.0, 50.0]},
    "RandomForest": {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [10, 15, 20],
        "estimator__min_samples_leaf": [3, 5],
    },
    "GradientBoosting": {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [4, 6, 8],
        "estimator__learning_rate": [0.03, 0.05, 0.1],
    },
}


def _build_estimator(model_name, params, available_features=None):
    """构建 estimator 或 Pipeline"""
    config = MODEL_CONFIGS[model_name]
    if config["need_scaler"]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", config["class"](**params)),
        ])
        return pipe
    else:
        return config["class"](**params)


# ========== 核心函数 ==========

def train_model(df, features, model_name="GradientBoosting",
                use_log_target=True, **kwargs):
    """
    训练单个模型

    Args:
        df: 训练数据
        features: 特征列名
        model_name: 模型名称
        use_log_target: 是否对 y 做 log1p 变换（推荐，因为房价右偏）
        **kwargs: 覆盖默认超参数

    Returns:
        (model, X_train, y_train_raw, feature_names, log_used)
    """
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["GradientBoosting"])
    params = config["params"].copy()
    params.update(kwargs)

    train_data = df.dropna(subset=features + ["unit_price"])
    available_features = [f for f in features if f in train_data.columns]

    X_train = train_data[available_features]
    y_train_raw = train_data["unit_price"].values

    if use_log_target:
        y_train = np.log1p(y_train_raw)
    else:
        y_train = y_train_raw

    model = _build_estimator(model_name, params, available_features)
    model.fit(X_train, y_train)

    # 注入特征名
    if isinstance(model, Pipeline):
        model.named_steps["estimator"].feature_names_in_ = np.array(available_features)

    return model, X_train, y_train_raw, available_features, use_log_target


def evaluate_model(model, X_test, y_test_raw, use_log_target=True):
    """评估模型"""
    final_est = model.named_steps["estimator"] if isinstance(model, Pipeline) else model

    if hasattr(final_est, "feature_names_in_"):
        required = list(final_est.feature_names_in_)
    else:
        required = [c for c in X_test.columns if not c.startswith("_")]

    # 用 numpy 数组避免列名/顺序问题
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]  # 去重
    X_sub = X_test[[c for c in required if c in X_test.columns]]
    X_arr = X_sub.fillna(0).values.astype(np.float64)

    y_vals = y_test_raw.values.astype(np.float64)
    mask = ~np.isnan(y_vals)
    X_arr, y_true = X_arr[mask], y_vals[mask]

    if len(X_arr) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

    y_pred_t = model.predict(X_arr)
    y_pred = np.expm1(np.clip(y_pred_t, -50, 50)) if use_log_target else np.clip(y_pred_t, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
            "y_test": pd.Series(y_true), "y_pred": y_pred, "n_samples": len(y_true)}


def cross_validate_model(df, features, model_name="GradientBoosting",
                         n_splits=3, use_log_target=True, **kwargs):
    """时序交叉验证"""
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["GradientBoosting"])
    params = config["params"].copy()
    params.update(kwargs)

    data = df.dropna(subset=features + ["unit_price", "year"]).copy()
    available_features = [f for f in features if f in data.columns]
    data = data.sort_values("year")
    years = sorted(data["year"].unique())

    if len(years) < n_splits + 1:
        n_splits = max(1, len(years) - 1)

    r2_scores, mae_scores = [], []
    fold_details = []

    for fold in range(n_splits):
        split_idx = len(years) - n_splits + fold
        train_years = years[:split_idx]
        val_year = years[split_idx]

        train_mask = data["year"].isin(train_years)
        val_mask = data["year"] == val_year

        X_tr = data.loc[train_mask, available_features]
        y_tr = data.loc[train_mask, "unit_price"].values
        X_val = data.loc[val_mask, available_features]
        y_val = data.loc[val_mask, "unit_price"].values

        if len(X_tr) < 50 or len(X_val) < 10:
            continue

        if use_log_target:
            y_tr_t = np.log1p(y_tr)
        else:
            y_tr_t = y_tr

        model = _build_estimator(model_name, params, available_features)
        model.fit(X_tr, y_tr_t)
        if isinstance(model, Pipeline):
            model.named_steps["estimator"].feature_names_in_ = np.array(available_features)

        y_pred_t = model.predict(X_val)
        y_pred = np.expm1(y_pred_t) if use_log_target else y_pred_t
        y_pred = np.clip(y_pred, 0, None)

        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2_scores.append(r2)
        mae_scores.append(mae)
        fold_details.append({
            "fold": fold + 1,
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "val_year": val_year,
            "train_size": len(X_tr), "val_size": len(X_val),
            "R²": round(r2, 4), "MAE": round(mae, 1),
        })

    if not r2_scores:
        return {"mean_r2": np.nan, "std_r2": np.nan, "mean_mae": np.nan,
                "std_mae": np.nan, "fold_details": []}

    return {
        "mean_r2": np.mean(r2_scores), "std_r2": np.std(r2_scores),
        "mean_mae": np.mean(mae_scores), "std_mae": np.std(mae_scores),
        "fold_details": fold_details,
    }


def train_all_models(df, features, use_log_target=True, **kwargs):
    """一键训练全部模型"""
    models, train_info, cv_info = {}, [], []
    for name, config in MODEL_CONFIGS.items():
        model, X_tr, y_tr, feats, log_used = train_model(
            df, features, name, use_log_target=use_log_target, **kwargs)
        models[name] = model
        train_info.append({
            "模型": config["label"], "Scaler": "是" if config["need_scaler"] else "否",
            "Log目标": "是" if log_used else "否",
        })
        cv = cross_validate_model(df, features, name, use_log_target=use_log_target, **kwargs)
        cv_info.append({
            "模型": config["label"],
            "CV-R²": round(cv["mean_r2"], 4) if not np.isnan(cv["mean_r2"]) else np.nan,
            "CV-R²_std": round(cv["std_r2"], 4) if not np.isnan(cv["std_r2"]) else np.nan,
            "CV-MAE": round(cv["mean_mae"], 1) if not np.isnan(cv["mean_mae"]) else np.nan,
        })
    return {"models": models, "train_info": pd.DataFrame(train_info),
            "cv_scores": pd.DataFrame(cv_info)}


def tune_hyperparameters(df, features, model_name, param_grid=None, cv=2, **kwargs):
    """GridSearchCV 超参数调优"""
    if model_name not in MODEL_CONFIGS:
        return train_model(df, features, model_name, **kwargs)

    config = MODEL_CONFIGS[model_name]
    grid = param_grid or PARAM_GRIDS.get(model_name, {})

    if not grid:
        return train_model(df, features, model_name, **kwargs)

    train_data = df.dropna(subset=features + ["unit_price", "year"])
    available_features = [f for f in features if f in train_data.columns]
    X = train_data[available_features]
    y = np.log1p(train_data["unit_price"].values)

    tscv = TimeSeriesSplit(n_splits=cv)
    base = _build_estimator(model_name, config["params"], available_features)

    search = GridSearchCV(base, grid, cv=tscv, scoring="neg_mean_absolute_error",
                          n_jobs=-1, verbose=0)
    search.fit(X, y)

    best_params = {k.replace("estimator__", ""): v for k, v in search.best_params_.items()}
    merged = config["params"].copy()
    merged.update(best_params)

    return train_model(df, features, model_name, use_log_target=True, **merged)


# ========== 辅助函数 ==========

def evaluate_by_category(df, features, model, category_col, category_labels,
                         use_log_target=True):
    """按类别评估"""
    results = []
    valid_data = df.dropna(subset=features + ["unit_price"])
    for cat_value, cat_label in category_labels.items():
        subset = valid_data[valid_data[category_col] == cat_value]
        if len(subset) < 10:
            continue
        available_features = [f for f in features if f in subset.columns]
        ev = evaluate_model(model, subset[available_features],
                            subset["unit_price"], use_log_target)
        results.append({
            "房源类型": cat_label, "MAE": ev["MAE"],
            "MAPE": ev.get("MAPE", np.nan), "R²": ev["R2"],
            "样本数": len(subset),
        })
    return pd.DataFrame(results)


def get_feature_importance(model, feature_names=None):
    """获取特征重要性"""
    final = model.named_steps["estimator"] if isinstance(model, Pipeline) else model

    if hasattr(final, "feature_importances_"):
        imp = final.feature_importances_
        imp_type = "feature_importances"
    elif hasattr(final, "coef_"):
        imp = np.abs(final.coef_).flatten()
        imp_type = "|coef| (标准化后)"
    else:
        return pd.DataFrame(), ""

    if hasattr(final, "feature_names_in_"):
        feature_names = list(final.feature_names_in_)
    elif feature_names is None:
        feature_names = [f"x{i}" for i in range(len(imp))]

    if len(imp) != len(feature_names):
        return pd.DataFrame(), ""

    df_imp = pd.DataFrame({"特征": feature_names, "重要性": imp}).sort_values("重要性", ascending=False)
    return df_imp.reset_index(drop=True), imp_type


def predict_price(model, input_dict, feature_names, use_log_target=True):
    """价格预估"""
    input_row = {f: input_dict.get(f, 0) for f in feature_names}
    input_df = pd.DataFrame([input_row])
    final = model.named_steps["estimator"] if isinstance(model, Pipeline) else model
    if hasattr(final, "feature_names_in_"):
        input_df = input_df[final.feature_names_in_]
    pred = model.predict(input_df)[0]
    return max(np.expm1(pred) if use_log_target else pred, 0)


def find_prediction_errors(df, features, model, top_n=10, use_log_target=True):
    """找误差最大的记录"""
    valid = df.dropna(subset=features + ["unit_price"])
    final_est = model.named_steps["estimator"] if isinstance(model, Pipeline) else model
    if hasattr(final_est, "feature_names_in_"):
        required = [c for c in final_est.feature_names_in_ if c in valid.columns]
    else:
        required = [c for c in features if c in valid.columns]
    X_arr = valid[required].fillna(0).values.astype(np.float64)
    y_actual = valid["unit_price"].values.astype(np.float64)
    y_pred_t = model.predict(X_arr)
    y_pred = np.expm1(np.clip(y_pred_t, -50, 50)) if use_log_target else np.clip(y_pred_t, 0, None)

    errors = pd.DataFrame({
        "index": valid.index,
        "实际单价": y_actual, "预测单价": y_pred,
        "绝对误差": np.abs(y_actual - y_pred),
        "误差%": np.abs((y_actual - y_pred) / np.clip(y_actual, 1, None)) * 100,
    })
    result = errors.nlargest(top_n, "绝对误差").copy()
    info_cols = ["town", "flat_type", "street_name", "storey_range",
                 "floor_area_sqm", "remaining_years", "flat_age", "resale_price"]
    for col in info_cols:
        if col in valid.columns:
            result[col] = valid.loc[result["index"], col].values
    return result.reset_index(drop=True)


def prepare_model_data(df):
    """按时间切分"""
    train_df = df[(df["year"] >= 2020) & (df["year"] <= 2023)].copy()
    test_df = df[df["year"] >= 2024].copy()
    return train_df, test_df
