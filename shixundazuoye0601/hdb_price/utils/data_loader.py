"""
数据加载与清洗模块
负责加载 HDB 转售数据、配套数据和辅助数据，并进行预处理
"""
import pandas as pd
import numpy as np
from pathlib import Path


def get_data_dir():
    """获取数据目录路径"""
    return Path(__file__).parent.parent / "data"


def load_hdb_data(filepath=None):
    """
    加载 HDB 转售成交数据并进行清洗

    Args:
        filepath: CSV 文件路径，默认从 data 目录加载

    Returns:
        pd.DataFrame: 清洗后的数据
    """
    if filepath is None:
        filepath = get_data_dir() / "hdb_resale.csv"

    df = pd.read_csv(filepath)

    # ========== 数据清洗 ==========

    # 1. 提取月份中的年份和年月信息
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["year"] = df["month"].dt.year
    df["year_month"] = df["month"].dt.to_period("M")

    # 2. 过滤 2020 年至今的数据
    df = df[df["year"] >= 2020].copy()

    # 3. 计算衍生字段
    df["unit_price"] = df["resale_price"] / df["floor_area_sqm"]

    # 4. 从 storey_range 提取楼层中位数
    df["storey_mid"] = df["storey_range"].apply(_extract_storey_mid)

    # 5. 从 remaining_lease 提取剩余租约年限
    df["remaining_years"] = df["remaining_lease"].apply(_extract_remaining_years)

    # 6. 计算房龄
    current_year = 2026
    df["flat_age"] = current_year - df["lease_commence_date"]

    # 7. 提取成交年份
    df["year"] = df["year"].astype(int)

    # 8. 处理缺失值
    df = df.dropna(
        subset=["floor_area_sqm", "resale_price", "remaining_years", "storey_mid"]
    )

    # 9. 统一镇区名称为大写
    df["town"] = df["town"].str.upper().str.strip()

    return df.reset_index(drop=True)


def _extract_storey_mid(storey_range):
    """
    从楼层范围字符串提取中位数
    例如 "07 TO 09" → 8, "01 TO 03" → 2
    """
    try:
        parts = str(storey_range).split(" TO ")
        low = int(parts[0])
        high = int(parts[1])
        return (low + high) / 2
    except (ValueError, IndexError):
        return np.nan


def _extract_remaining_years(remaining_lease):
    """
    从剩余租约字符串提取年限数值
    例如 "63 years 06 months" → 63.5, "61 years 04 months" → 61.33
    """
    try:
        s = str(remaining_lease)
        years = 0
        months = 0
        if "years" in s:
            years = int(s.split(" years")[0].strip())
        if "months" in s or "month" in s:
            parts = s.replace("years", "").replace("year", "").strip()
            months = int(parts.split(" ")[0])
        return years + months / 12.0
    except (ValueError, IndexError, AttributeError):
        return np.nan


def load_town_locations(filepath=None):
    """加载镇区中心点坐标"""
    if filepath is None:
        filepath = get_data_dir() / "town_locations.csv"
    df = pd.read_csv(filepath)
    df["town"] = df["town"].str.upper().str.strip()
    return df


def load_mrt_stations(filepath=None):
    """加载 MRT/LRT 站点数据"""
    if filepath is None:
        filepath = get_data_dir() / "mrt_stations.csv"
    return pd.read_csv(filepath)


def load_schools(filepath=None):
    """加载学校数据"""
    if filepath is None:
        filepath = get_data_dir() / "schools.csv"
    df = pd.read_csv(filepath)
    df["town"] = df["town"].str.upper().str.strip()
    return df


def load_events(filepath=None):
    """加载政策事件数据"""
    if filepath is None:
        filepath = get_data_dir() / "events.csv"
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_filtered_data(df, towns=None, year_range=None, flat_type=None, area_range=None):
    """
    根据筛选条件过滤数据

    Args:
        df: 完整数据 DataFrame
        towns: 镇区列表
        year_range: 年份范围 tuple (min, max)
        flat_type: 房型选择（"不限"表示不过滤）
        area_range: 面积范围 tuple (min, max)

    Returns:
        pd.DataFrame: 筛选后的数据
    """
    filtered = df.copy()

    if towns:
        filtered = filtered[filtered["town"].isin(towns)]

    if year_range:
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    if flat_type and flat_type != "不限":
        filtered = filtered[filtered["flat_type"] == flat_type]

    if area_range:
        filtered = filtered[
            (filtered["floor_area_sqm"] >= area_range[0]) &
            (filtered["floor_area_sqm"] <= area_range[1])
        ]

    return filtered


def get_stats(df):
    """
    计算基本统计指标

    Returns:
        dict: 包含各种统计值的字典
    """
    stats = {
        "count": len(df),
        "avg_unit_price": df["unit_price"].mean(),
        "avg_total_price": df["resale_price"].mean(),
        "min_unit_price": df["unit_price"].min(),
        "max_unit_price": df["unit_price"].max(),
        "median_unit_price": df["unit_price"].median(),
        "std_unit_price": df["unit_price"].std(),
        "avg_area": df["floor_area_sqm"].mean(),
        "avg_remaining_years": df["remaining_years"].mean(),
        "avg_flat_age": df["flat_age"].mean(),
    }
    return stats
