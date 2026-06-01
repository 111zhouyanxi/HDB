"""
模型训练与预测模块
负责房价预测模型的训练、评估、对比和价格预估

优化内容:
- StandardScaler: 线性模型自动标准化特征
- TimeSeriesSplit: 时序交叉验证，评估模型稳定性
- 多模型对比: 一键训练所有模型并横向对比
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ========== 模型配置 ==========
# need_scaler=True 表示该模型需要特征标准化
MODEL_CONFIGS = {
    "LinearRegression": {
        "class": LinearRegression,
        "label": "线性回归",
        "params": {},
        "need_scaler": True,
        "description": "最简单的线性模型，可解释性强",
    },
    "Ridge": {
        "class": Ridge,
        "label": "岭回归",
        "params": {"alpha": 1.0},
        "need_scaler": True,
        "description": "带 L2 正则化的线性模型，防止过拟合",
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "label": "随机森林",
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        "need_scaler": False,
        "description": "集成多棵决策树，捕捉非线性关系",
    },
    "GradientBoosting": {
        "class": GradientBoostingRegressor,
        "label": "梯度提升树",
        "params": {"n_estimators": 100, "max_depth": 5,
                   "learning_rate": 0.1, "random_state": 42},
        "need_scaler": False,
        "description": "逐步优化残差，预测精度高",
    },
}


# ========== 核心训练函数 ==========

def train_model(df, features, model_name="RandomForest", **kwargs):
    """
    训练单个预测模型（线性模型自动带 StandardScaler Pipeline）

    Args:
        df: 包含特征和目标的数据
        features: 特征列名列表
        model_name: 模型名称
        **kwargs: 模型超参数

    Returns:
        tuple: (model_or_pipeline, X_train, y_train, feature_names)
    """
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["RandomForest"])

    # 构建参数
    params = config["params"].copy()
    params.update(kwargs)

    # 准备数据
    train_data = df.dropna(subset=features + ["unit_price"])
    available_features = [f for f in features if f in train_data.columns]

    X_train = train_data[available_features]
    y_train = train_data["unit_price"]

    # 线性模型 → 用 Pipeline(StandardScaler + Model)
    if config["need_scaler"]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", config["class"](**params)),
        ])
        model.fit(X_train, y_train)
        # StandardScaler 会将 DataFrame 转为 numpy array，导致 feature_names_in_ 丢失
        # 手动注入特征名到内部 estimator
        model.named_steps["estimator"].feature_names_in_ = np.array(available_features)
    else:
        model = config["class"](**params)
        model.fit(X_train, y_train)

    return model, X_train, y_train, available_features


def evaluate_model(model, X_test, y_test):
    """
    评估模型表现（自动处理 Pipeline 和特征对齐）

    Args:
        model: 已训练的模型（或 Pipeline）
        X_test: 测试特征
        y_test: 测试目标

    Returns:
        dict: 包含 MAE, RMSE, R², MAPE 等指标
    """
    # 获取模型需要的特征名
    final_estimator = model
    if isinstance(model, Pipeline):
        final_estimator = model.named_steps["estimator"]

    if hasattr(final_estimator, "feature_names_in_"):
        required_features = list(final_estimator.feature_names_in_)
        # 确保所有必需特征都存在
        missing = [f for f in required_features if f not in X_test.columns]
        if missing:
            for f in missing:
                X_test = X_test.copy()
                X_test[f] = 0
        X_test = X_test[required_features].copy()
    else:
        X_test = X_test.dropna()

    X_test = X_test.dropna()

    # 使用非 NaN 的行
    valid_idx = X_test.index.intersection(y_test.dropna().index)
    X_test = X_test.loc[valid_idx]
    y_test = y_test.loc[valid_idx]

    if len(X_test) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "MAPE": np.nan, "n_samples": 0}

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "y_test": y_test,
        "y_pred": y_pred,
        "n_samples": len(X_test),
    }


def cross_validate_model(df, features, model_name="RandomForest",
                         n_splits=3, **kwargs):
    """
    使用时序交叉验证评估模型稳定性

    按年份顺序切分，每次用前 N 年训练，后 1 年验证。
    例如 n_splits=3 时:
      Fold 1: train=2020-2021, val=2022
      Fold 2: train=2020-2022, val=2023
      Fold 3: train=2020-2023, val=2024

    Args:
        df: 包含特征和目标的数据（需有 year 列）
        features: 特征列名列表
        model_name: 模型名称
        n_splits: 切分折数
        **kwargs: 模型超参数

    Returns:
        dict: {mean_r2, std_r2, mean_mae, std_mae, scores, fold_details}
    """
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["RandomForest"])
    params = config["params"].copy()
    params.update(kwargs)

    # 准备数据
    data = df.dropna(subset=features + ["unit_price", "year"]).copy()
    available_features = [f for f in features if f in data.columns]

    # 按年份排序
    data = data.sort_values("year")
    years = sorted(data["year"].unique())

    if len(years) < n_splits + 1:
        # 年份不够，减少折数
        n_splits = max(1, len(years) - 1)

    r2_scores = []
    mae_scores = []
    fold_details = []

    for fold in range(n_splits):
        # 训练年份：从最早到倒数第 (n_splits - fold) 年
        split_idx = len(years) - n_splits + fold
        train_years = years[:split_idx]
        val_year = years[split_idx]

        train_mask = data["year"].isin(train_years)
        val_mask = data["year"] == val_year

        X_tr = data.loc[train_mask, available_features]
        y_tr = data.loc[train_mask, "unit_price"]
        X_val = data.loc[val_mask, available_features]
        y_val = data.loc[val_mask, "unit_price"]

        if len(X_tr) < 50 or len(X_val) < 10:
            continue

        # 训练
        if config["need_scaler"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("estimator", config["class"](**params)),
            ])
            model.fit(X_tr, y_tr)
            model.named_steps["estimator"].feature_names_in_ = np.array(available_features)
        else:
            model = config["class"](**params)
            model.fit(X_tr, y_tr)

        # 评估
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        r2_scores.append(r2)
        mae_scores.append(mae)
        fold_details.append({
            "fold": fold + 1,
            "train_years": f"{train_years[0]}-{train_years[-1]}",
            "val_year": val_year,
            "train_size": len(X_tr),
            "val_size": len(X_val),
            "R²": round(r2, 4),
            "MAE": round(mae, 1),
        })

    if len(r2_scores) == 0:
        return {
            "mean_r2": np.nan, "std_r2": np.nan,
            "mean_mae": np.nan, "std_mae": np.nan,
            "scores": [], "fold_details": [],
        }

    return {
        "mean_r2": np.mean(r2_scores),
        "std_r2": np.std(r2_scores),
        "mean_mae": np.mean(mae_scores),
        "std_mae": np.std(mae_scores),
        "r2_scores": r2_scores,
        "mae_scores": mae_scores,
        "fold_details": fold_details,
    }


def train_all_models(df, features, **kwargs):
    """
    一键训练所有模型并返回横向对比结果

    Args:
        df: 训练数据（2020-2023）
        features: 特征列名列表
        **kwargs: 可覆盖默认超参数

    Returns:
        dict: {
            models: {name: trained_model},
            train_scores: DataFrame (单次拟合的指标对比),
            cv_scores: DataFrame (交叉验证的指标对比),
        }
    """
    models = {}
    train_results = []
    cv_results = []

    for name, config in MODEL_CONFIGS.items():
        # 训练模型
        model, X_tr, y_tr, feats = train_model(df, features, name, **kwargs)
        models[name] = model

        # 在训练集上的交叉验证
        cv = cross_validate_model(df, features, name, n_splits=3, **kwargs)

        train_results.append({
            "模型": config["label"],
            "模型名称": name,
            "需要标准化": "是" if config["need_scaler"] else "否",
            "描述": config["description"],
        })
        cv_results.append({
            "模型": config["label"],
            "CV-R²均值": round(cv["mean_r2"], 4) if not np.isnan(cv["mean_r2"]) else np.nan,
            "CV-R²标准差": round(cv["std_r2"], 4) if not np.isnan(cv["std_r2"]) else np.nan,
            "CV-MAE均值": round(cv["mean_mae"], 1) if not np.isnan(cv["mean_mae"]) else np.nan,
            "CV-MAE标准差": round(cv["std_mae"], 1) if not np.isnan(cv["std_mae"]) else np.nan,
        })

    return {
        "models": models,
        "train_info": pd.DataFrame(train_results),
        "cv_scores": pd.DataFrame(cv_results),
    }


# ========== 评估辅助函数 ==========

def evaluate_by_category(df, features, model, category_col, category_labels):
    """
    按类别分组评估模型表现
    """
    results = []
    valid_data = df.dropna(subset=features + ["unit_price"])

    for cat_value, cat_label in category_labels.items():
        subset = valid_data[valid_data[category_col] == cat_value]
        if len(subset) < 10:
            continue

        available_features = [f for f in features if f in subset.columns]
        X_sub = subset[available_features]
        y_sub = subset["unit_price"]

        eval_result = evaluate_model(model, X_sub, y_sub)

        results.append({
            "房源类型": cat_label,
            "MAE": eval_result["MAE"],
            "MAPE": eval_result.get("MAPE", np.nan),
            "R²": eval_result["R2"],
            "样本数": len(subset),
        })

    return pd.DataFrame(results)


def get_feature_importance(model, feature_names=None):
    """
    获取特征重要性排名（自动识别线性模型和树模型）
    """
    # 处理 Pipeline
    final_model = model
    if isinstance(model, Pipeline):
        final_model = model.named_steps["estimator"]

    if hasattr(final_model, "feature_importances_"):
        importance = final_model.feature_importances_
        imp_type = "feature_importances"
    elif hasattr(final_model, "coef_"):
        importance = np.abs(final_model.coef_)
        if len(importance.shape) > 1:
            importance = importance.flatten()
        imp_type = "|coef| (已标准化)"
    else:
        return pd.DataFrame(columns=["特征", "重要性"]), ""

    # 对齐特征名
    if hasattr(final_model, "feature_names_in_"):
        feature_names = list(final_model.feature_names_in_)
    elif feature_names is None:
        feature_names = [f"feat_{i}" for i in range(len(importance))]

    if len(importance) != len(feature_names):
        return pd.DataFrame(columns=["特征", "重要性"]), ""

    imp_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": importance,
    }).sort_values("重要性", ascending=False).reset_index(drop=True)

    return imp_df, imp_type


def predict_price(model, input_dict, feature_names):
    """
    使用训练好的模型预估房价
    """
    # 构建输入向量
    input_row = {}
    for feat in feature_names:
        input_row[feat] = input_dict.get(feat, 0)

    input_df = pd.DataFrame([input_row])

    # 按模型特征顺序排列
    final_model = model
    if isinstance(model, Pipeline):
        final_model = model.named_steps["estimator"]

    if hasattr(final_model, "feature_names_in_"):
        input_df = input_df[final_model.feature_names_in_]

    predicted_unit_price = model.predict(input_df)[0]
    return max(predicted_unit_price, 0)


def find_prediction_errors(df, features, model, top_n=10):
    """
    找出预测误差最大的记录
    """
    valid_data = df.dropna(subset=features + ["unit_price"])
    available_features = [f for f in features if f in valid_data.columns]

    X = valid_data[available_features]
    y_actual = valid_data["unit_price"]
    y_pred = model.predict(X)

    errors = pd.DataFrame({
        "index": valid_data.index,
        "实际单价": y_actual.values,
        "预测单价": y_pred,
        "绝对误差": np.abs(y_actual.values - y_pred),
        "误差百分比": np.abs((y_actual.values - y_pred) / y_actual.values) * 100,
    })

    # 合并原始信息
    result = errors.nlargest(top_n, "绝对误差").copy()
    info_cols = ["town", "flat_type", "street_name", "storey_range",
                 "floor_area_sqm", "remaining_years", "flat_age", "resale_price"]
    for col in info_cols:
        if col in valid_data.columns:
            result[col] = valid_data.loc[result["index"], col].values

    return result.reset_index(drop=True)


def prepare_model_data(df):
    """
    按时间划分训练集和测试集
    """
    train_df = df[(df["year"] >= 2020) & (df["year"] <= 2023)].copy()
    test_df = df[df["year"] >= 2024].copy()
    return train_df, test_df
