"""
模型训练与预测模块
负责房价预测模型的训练、评估、对比和价格预估
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# 可用的模型配置
MODEL_CONFIGS = {
    "LinearRegression": {
        "class": LinearRegression,
        "label": "线性回归",
        "params": {},
        "description": "最简单的线性模型，可解释性强"
    },
    "Ridge": {
        "class": Ridge,
        "label": "岭回归",
        "params": {"alpha": 1.0},
        "description": "带 L2 正则化的线性模型，防止过拟合"
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "label": "随机森林",
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "description": "集成多棵决策树，捕捉非线性关系"
    },
    "GradientBoosting": {
        "class": GradientBoostingRegressor,
        "label": "梯度提升树",
        "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
        "description": "逐步优化残差，预测精度高"
    },
}


def train_model(df, features, model_name="RandomForest", **kwargs):
    """
    训练预测模型

    Args:
        df: 包含特征和目标的数据
        features: 特征列名列表
        model_name: 模型名称
        **kwargs: 模型超参数

    Returns:
        tuple: (model, X_train, y_train, feature_names)
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

    # 训练模型
    model = config["class"](**params)
    model.fit(X_train, y_train)

    return model, X_train, y_train, available_features


def evaluate_model(model, X_test, y_test):
    """
    评估模型表现

    Args:
        model: 已训练的模型
        X_test: 测试特征
        y_test: 测试目标

    Returns:
        dict: 包含 MAE, RMSE, R², MAPE 等指标
    """
    # 只使用模型训练时的特征，按模型需要的顺序排列
    if hasattr(model, "feature_names_in_"):
        required_features = list(model.feature_names_in_)
        # 确保所有必需特征都存在
        missing = [f for f in required_features if f not in X_test.columns]
        if missing:
            # 为缺失的列补 0
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
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

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


def evaluate_by_category(df, features, model, category_col, category_labels):
    """
    按类别分组评估模型表现

    Args:
        df: 测试数据
        features: 特征列名
        model: 已训练模型
        category_col: 分类列名
        category_labels: 各类别的标签映射

    Returns:
        pd.DataFrame: 各类别的评估结果
    """
    results = []
    valid_data = df.dropna(subset=features + ["unit_price"])
    available_features = [f for f in features if f in valid_data.columns]

    for cat_value, cat_label in category_labels.items():
        subset = valid_data[valid_data[category_col] == cat_value]
        if len(subset) < 10:
            continue

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


def get_feature_importance(model, feature_names):
    """
    获取特征重要性排名

    Args:
        model: 已训练的模型
        feature_names: 特征名称列表

    Returns:
        pd.DataFrame: 特征重要性表
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1:
            importance = importance.flatten()
    else:
        return pd.DataFrame()

    # 对齐特征名和重要性
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    if len(importance) != len(feature_names):
        return pd.DataFrame()

    imp_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": importance,
    }).sort_values("重要性", ascending=False)

    return imp_df


def predict_price(model, input_dict, feature_names):
    """
    使用训练好的模型预估房价

    Args:
        model: 已训练的模型
        input_dict: 用户输入的特征值
        feature_names: 模型训练时的特征名列表

    Returns:
        float: 预估单价（新币/㎡）
    """
    # 构建输入向量
    input_row = {}
    for feat in feature_names:
        input_row[feat] = input_dict.get(feat, 0)

    input_df = pd.DataFrame([input_row])
    # 按模型特征顺序排列
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    predicted_unit_price = model.predict(input_df)[0]
    return max(predicted_unit_price, 0)  # 确保非负


def find_prediction_errors(df, features, model, top_n=10):
    """
    找出预测误差最大的记录

    Args:
        df: 测试数据
        features: 特征列名
        model: 已训练模型
        top_n: 返回误差最大的 N 条记录

    Returns:
        pd.DataFrame: 误差最大的记录
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

    Args:
        df: 完整数据

    Returns:
        tuple: (train_df, test_df)
    """
    train_df = df[(df["year"] >= 2020) & (df["year"] <= 2023)].copy()
    test_df = df[df["year"] >= 2024].copy()
    return train_df, test_df
