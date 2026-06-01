"""
地理工具模块
提供距离计算、最近配套查找、镇区级配套统计等功能
"""
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine(lat1, lon1, lat2, lon2):
    """
    使用 Haversine 公式计算两点间的地表距离（公里）

    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度

    Returns:
        float: 两点间的距离（公里）
    """
    R = 6371  # 地球半径（公里）
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def find_nearest_distance(row_lat, row_lon, ref_df, lat_col="latitude", lon_col="longitude"):
    """
    计算某点到参考数据集中最近点的距离

    Args:
        row_lat, row_lon: 目标点的经纬度
        ref_df: 参考数据集（含经纬度列）
        lat_col, lon_col: 经纬度列名

    Returns:
        float: 到最近点的距离（公里），若无参考数据则返回 NaN
    """
    if ref_df.empty:
        return np.nan
    distances = [
        haversine(row_lat, row_lon, r[lat_col], r[lon_col])
        for _, r in ref_df.iterrows()
    ]
    return min(distances) if distances else np.nan


def count_nearby_facilities(row_lat, row_lon, ref_df, radius_km=1.0,
                            lat_col="latitude", lon_col="longitude"):
    """
    统计某点半径范围内的参考设施数量

    Args:
        row_lat, row_lon: 目标点的经纬度
        ref_df: 参考数据集
        radius_km: 半径（公里）
        lat_col, lon_col: 经纬度列名

    Returns:
        int: 范围内的设施数量
    """
    if ref_df.empty:
        return 0
    count = 0
    for _, r in ref_df.iterrows():
        dist = haversine(row_lat, row_lon, r[lat_col], r[lon_col])
        if dist <= radius_km:
            count += 1
    return count


def compute_town_distances(town_df, mrt_df, schools_df):
    """
    为每条镇区成交记录计算配套距离（镇区级近似）

    Args:
        town_df: 含 town 列的成交数据（或按镇区聚合后的数据）
        mrt_df: MRT 站点数据
        schools_df: 学校数据

    Returns:
        pd.DataFrame: 增加了 mrt_distance, school_distance, mrt_count, school_count 列的数据
    """
    result = town_df.copy()
    result["mrt_distance"] = np.nan
    result["school_distance"] = np.nan
    result["mrt_count"] = 0
    result["school_count"] = 0

    if "latitude" not in result.columns or "longitude" not in result.columns:
        return result

    for idx, row in result.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        if pd.notna(lat) and pd.notna(lon):
            result.at[idx, "mrt_distance"] = find_nearest_distance(lat, lon, mrt_df)
            result.at[idx, "school_distance"] = find_nearest_distance(lat, lon, schools_df)
            result.at[idx, "mrt_count"] = count_nearby_facilities(lat, lon, mrt_df, 1.0)
            result.at[idx, "school_count"] = count_nearby_facilities(lat, lon, schools_df, 1.0)

    return result


def merge_town_coords(df, town_locations_df):
    """
    将镇区中心点坐标合并到成交数据中

    Args:
        df: HDB 成交数据
        town_locations_df: 镇区坐标数据

    Returns:
        pd.DataFrame: 合并了坐标的数据
    """
    return df.merge(
        town_locations_df[["town", "latitude", "longitude", "estate_type"]],
        on="town",
        how="left"
    )


def classify_mrt_proximity(distance):
    """
    根据距离将 MRT 邻近程度分类

    Args:
        distance: 到 MRT 的距离（公里）

    Returns:
        str: 分类标签
    """
    if pd.isna(distance):
        return "未知"
    if distance <= 0.5:
        return "MRT沿线 (<500m)"
    elif distance <= 1.0:
        return "近距离 (500m-1km)"
    else:
        return "远离MRT (>1km)"
