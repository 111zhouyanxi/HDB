"""
特征工程模块
将原始字段转化为模型可用的特征
"""
import pandas as pd
import numpy as np


# 成熟组屋区列表
MATURE_ESTATES = [
    "QUEENSTOWN", "TOA PAYOH", "ANG MO KIO", "BEDOK",
    "TAMPINES", "CLEMENTI", "BISHAN", "BUKIT MERAH",
    "BUKIT TIMAH", "CENTRAL AREA", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "SERANGOON"
]


def build_features(df, town_locations_df=None, mrt_df=None, schools_df=None):
    """
    构建模型特征矩阵

    Args:
        df: HDB 成交数据（已清洗）
        town_locations_df: 镇区坐标数据（可选，用于合并 estate_type）
        mrt_df: MRT 站点数据（可选）
        schools_df: 学校数据（可选）

    Returns:
        pd.DataFrame: 特征矩阵
        list: 特征列名列表
    """
    feats = df.copy()

    # 1. 数值特征：直接使用
    feats["floor_area_sqm"] = feats["floor_area_sqm"].astype(float)
    feats["remaining_years"] = feats["remaining_years"].astype(float)
    feats["flat_age"] = feats["flat_age"].astype(float)
    feats["storey_mid"] = feats["storey_mid"].astype(float)

    # 2. 房型编码
    flat_type_map = {
        "2 ROOM": 1, "3 ROOM": 2, "4 ROOM": 3,
        "5 ROOM": 4, "EXECUTIVE": 5, "MULTI-GENERATION": 5
    }
    feats["flat_type_code"] = feats["flat_type"].map(flat_type_map).fillna(3).astype(int)

    # 3. 成熟区标志
    if "estate_type" in feats.columns:
        feats["is_mature"] = (feats["estate_type"] == "Mature").astype(int)
    elif town_locations_df is not None:
        feats = feats.merge(
            town_locations_df[["town", "estate_type"]],
            on="town", how="left"
        )
        feats["is_mature"] = (feats["estate_type"] == "Mature").astype(int)
    else:
        feats["is_mature"] = feats["town"].isin(MATURE_ESTATES).astype(int)

    # 4. 楼层分组编码
    feats["storey_level"] = pd.cut(
        feats["storey_mid"],
        bins=[0, 6, 15, 100],
        labels=["low", "mid", "high"]
    )
    feats["is_high_floor"] = (feats["storey_mid"] >= 15).astype(int)
    feats["is_low_floor"] = (feats["storey_mid"] <= 3).astype(int)

    # 5. MRT 距离特征（如果提供了坐标）
    if "mrt_distance" in feats.columns:
        feats["near_mrt"] = (feats["mrt_distance"] <= 0.5).astype(int)
        feats["mrt_distance_km"] = feats["mrt_distance"].fillna(feats["mrt_distance"].median())
    else:
        feats["near_mrt"] = 0
        feats["mrt_distance_km"] = 2.0  # 默认中等距离

    # 6. 学校距离特征
    if "school_distance" in feats.columns:
        feats["near_school"] = (feats["school_distance"] <= 1.0).astype(int)
    else:
        feats["near_school"] = 0

    # 7. 交易年份特征
    if "year" in feats.columns:
        feats["year_feat"] = feats["year"].astype(int) - 2020  # 以 2020 为基准

    # 8. 镇区 One-Hot 编码（保留在特征中）
    town_dummies = pd.get_dummies(feats["town"], prefix="town")
    feats = pd.concat([feats, town_dummies], axis=1)

    # 9. 房型 One-Hot 编码
    flat_dummies = pd.get_dummies(feats["flat_type"], prefix="flat")
    feats = pd.concat([feats, flat_dummies], axis=1)

    # 10. 交互特征
    feats["area_per_room"] = feats["floor_area_sqm"] / (feats["flat_type_code"] + 1)
    feats["age_lease_interact"] = feats["flat_age"] * (99 - feats["remaining_years"].clip(upper=99))

    return feats


# 基础特征列名（用于模型训练）
BASE_FEATURES = [
    "floor_area_sqm",       # 建筑面积
    "remaining_years",      # 剩余租约年限
    "flat_age",             # 房龄
    "storey_mid",           # 楼层中位数
    "flat_type_code",       # 房型编码
    "is_mature",            # 是否成熟区
    "is_high_floor",        # 是否高楼层 (>=15)
    "is_low_floor",         # 是否低楼层 (<=3)
    "near_mrt",             # 是否接近MRT
    "mrt_distance_km",      # 到MRT距离
    "near_school",          # 是否接近学校
    "year_feat",            # 成交年份特征
    "area_per_room",        # 每房面积（面积/房型）
    "age_lease_interact",   # 房龄×已消耗租约交互
]


def get_feature_columns(df, include_town_dummies=True):
    """
    获取可用于模型训练的特征列名列表

    Args:
        df: 包含所有生成特征的数据
        include_town_dummies: 是否包含镇区 One-Hot 编码

    Returns:
        list: 模型可用的特征列名
    """
    features = list(BASE_FEATURES)

    # 检查哪些特征实际存在于数据中
    available = [f for f in features if f in df.columns]

    # 可选：添加 One-Hot 列
    if include_town_dummies:
        town_cols = [c for c in df.columns if c.startswith("town_") and c != "town_code"]
        available.extend(town_cols)

    # 添加房型 One-Hot
    flat_cols = [c for c in df.columns if c.startswith("flat_") and c not in ("flat_type_code", "flat_type", "flat_model", "flat_age")]
    available.extend(flat_cols)

    # 只保留数值类型列，排除字符串/对象类型
    numeric_available = []
    for col in available:
        if col in df.columns:
            dtype_str = str(df[col].dtype)
            if any(t in dtype_str for t in ('float', 'int', 'bool')):
                numeric_available.append(col)
            elif df[col].dtype.name == 'bool':
                numeric_available.append(col)

    return numeric_available


def get_categorical_features(df):
    """
    获取需要分类型分析的特征分组

    Returns:
        dict: 特征名 -> 分组映射
    """
    cats = {
        "flat_type": "房型",
        "storey_level": "楼层级别",
        "estate_type": "镇区类型",
        "mrt_proximity": "MRT邻近度",
    }
    result = {}
    for col, label in cats.items():
        if col in df.columns:
            result[col] = label
    return result


def classify_flat_category(flat_type, remaining_years, mrt_distance=None):
    """
    将房源分类为不同维度类型

    Args:
        flat_type: 房型
        remaining_years: 剩余租约
        mrt_distance: 到 MRT 距离

    Returns:
        dict: 各维度的分类结果
    """
    categories = {}

    # 按户型分类
    if flat_type in ["2 ROOM", "3 ROOM"]:
        categories["size_cat"] = "小户型"
    elif flat_type == "4 ROOM":
        categories["size_cat"] = "中户型"
    else:
        categories["size_cat"] = "大户型"

    # 按新旧分类
    if remaining_years < 60:
        categories["age_cat"] = "老旧组屋"
    elif remaining_years >= 80:
        categories["age_cat"] = "新近组屋"
    else:
        categories["age_cat"] = "中年组屋"

    # 按 MRT 距离分类
    if mrt_distance is not None and not pd.isna(mrt_distance):
        if mrt_distance <= 0.5:
            categories["mrt_cat"] = "MRT沿线"
        else:
            categories["mrt_cat"] = "远离MRT"
    else:
        categories["mrt_cat"] = "未知"

    return categories
