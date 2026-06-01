"""
分析函数模块
提供思考题分析、策略验证、事件冲击分析等分析功能
"""
import pandas as pd
import numpy as np
from .data_loader import get_filtered_data
from .feature_eng import MATURE_ESTATES


def analyze_town_price_trend(df, towns=None):
    """
    分析各镇区近5年的价格涨跌幅

    Args:
        df: 完整数据
        towns: 要分析的镇区列表

    Returns:
        pd.DataFrame: 各镇区的价格趋势数据
    """
    if towns:
        data = df[df["town"].isin(towns)]
    else:
        data = df

    # 按镇区和年份聚合计
    yearly = data.groupby(["town", "year"]).agg(
        avg_unit_price=("unit_price", "mean"),
        avg_total_price=("resale_price", "mean"),
        transaction_count=("resale_price", "count"),
        avg_remaining_years=("remaining_years", "mean"),
    ).reset_index()

    # 计算每个镇区的涨幅
    trends = []
    for town_name, group in yearly.groupby("town"):
        group = group.sort_values("year")
        if len(group) >= 2:
            first_year_price = group.iloc[0]["avg_unit_price"]
            last_year_price = group.iloc[-1]["avg_unit_price"]
            total_change = (last_year_price - first_year_price) / first_year_price * 100
            cagr = ((last_year_price / first_year_price) ** (1 / max(len(group) - 1, 1)) - 1) * 100

        trends.append({
            "town": town_name,
            "2020均价": group[group["year"] == 2020]["avg_unit_price"].values[0] if 2020 in group["year"].values else np.nan,
            "2024均价": group[group["year"] == 2024]["avg_unit_price"].values[0] if 2024 in group["year"].values else np.nan,
            "总涨幅(%)": total_change if len(group) >= 2 else np.nan,
            "年化涨幅(%)": cagr if len(group) >= 2 else np.nan,
            "总成交量": group["transaction_count"].sum(),
            "年平均剩余租约": group["avg_remaining_years"].mean(),
        })

    result = pd.DataFrame(trends)
    if len(result) > 0:
        result = result.sort_values("总涨幅(%)", ascending=False)
    return result


def analyze_preservation_value(df, category_col, category_value, label):
    """
    分析某类组屋的保值能力

    Args:
        df: 完整数据
        category_col: 分类列名
        category_value: 分类值
        label: 标签

    Returns:
        dict: 保值分析结果
    """
    subset = df[df[category_col] == category_value]
    other = df[df[category_col] != category_value]

    yearly_sub = subset.groupby("year")["unit_price"].agg(["mean", "std", "count"]).reset_index()
    yearly_other = other.groupby("year")["unit_price"].agg(["mean", "std", "count"]).reset_index()

    # 计算涨幅
    if len(yearly_sub) >= 2:
        sub_first = yearly_sub.iloc[0]["mean"]
        sub_last = yearly_sub.iloc[-1]["mean"]
        sub_change = (sub_last - sub_first) / sub_first * 100
    else:
        sub_change = np.nan

    if len(yearly_other) >= 2:
        other_first = yearly_other.iloc[0]["mean"]
        other_last = yearly_other.iloc[-1]["mean"]
        other_change = (other_last - other_first) / other_first * 100
    else:
        other_change = np.nan

    return {
        "类别": label,
        "类别涨幅(%)": sub_change,
        "对比组涨幅(%)": other_change,
        "差值(%)": sub_change - other_change if not (np.isnan(sub_change) or np.isnan(other_change)) else np.nan,
        "类别成交量": len(subset),
        "对比组成交量": len(other),
    }


def validate_strategy(df, strategy_config, test_df=None):
    """
    验证购房策略的表现

    Args:
        df: 策略形成期数据 (2020-2023)
        strategy_config: 策略配置字典
        test_df: 验证期数据 (2024+)

    Returns:
        dict: 策略验证结果
    """
    # 筛选策略组
    strategy_df = df.copy()

    if "towns" in strategy_config and strategy_config["towns"]:
        strategy_df = strategy_df[strategy_df["town"].isin(strategy_config["towns"])]

    if "flat_types" in strategy_config and strategy_config["flat_types"]:
        strategy_df = strategy_df[strategy_df["flat_type"].isin(strategy_config["flat_types"])]

    if "max_price" in strategy_config and strategy_config["max_price"]:
        strategy_df = strategy_df[strategy_df["resale_price"] <= strategy_config["max_price"]]

    if "min_remaining_years" in strategy_config and strategy_config["min_remaining_years"]:
        strategy_df = strategy_df[strategy_df["remaining_years"] >= strategy_config["min_remaining_years"]]

    if "max_mrt_distance" in strategy_config and strategy_config.get("max_mrt_distance"):
        if "mrt_distance" in strategy_df.columns:
            strategy_df = strategy_df[strategy_df["mrt_distance"] <= strategy_config["max_mrt_distance"]]

    # 基准组：全部符合户型和预算约束的组屋
    baseline_df = df.copy()
    if "towns" in strategy_config and strategy_config["towns"]:
        baseline_df = baseline_df[baseline_df["town"].isin(strategy_config["towns"])]
    if "flat_types" in strategy_config and strategy_config["flat_types"]:
        baseline_df = baseline_df[baseline_df["flat_type"].isin(strategy_config["flat_types"])]
    if "max_price" in strategy_config and strategy_config["max_price"]:
        baseline_df = baseline_df[baseline_df["resale_price"] <= strategy_config["max_price"]]

    result = {
        "strategy_count": len(strategy_df),
        "baseline_count": len(baseline_df),
        "strategy_avg_price": strategy_df["unit_price"].mean(),
        "baseline_avg_price": baseline_df["unit_price"].mean(),
        "strategy_avg_total": strategy_df["resale_price"].mean(),
        "baseline_avg_total": baseline_df["resale_price"].mean(),
    }

    # 验证期分析
    if test_df is not None and len(test_df) > 0:
        # 用相同条件筛选验证期数据
        test_strategy = test_df.copy()
        test_baseline = test_df.copy()

        if "towns" in strategy_config and strategy_config["towns"]:
            test_strategy = test_strategy[test_strategy["town"].isin(strategy_config["towns"])]
            test_baseline = test_baseline[test_baseline["town"].isin(strategy_config["towns"])]
        if "flat_types" in strategy_config and strategy_config["flat_types"]:
            test_strategy = test_strategy[test_strategy["flat_type"].isin(strategy_config["flat_types"])]
            test_baseline = test_baseline[test_baseline["flat_type"].isin(strategy_config["flat_types"])]
        if "max_price" in strategy_config and strategy_config["max_price"]:
            test_strategy = test_strategy[test_strategy["resale_price"] <= strategy_config["max_price"]]
            test_baseline = test_baseline[test_baseline["resale_price"] <= strategy_config["max_price"]]

        # 计算验证期指标
        strategy_yearly = test_strategy.groupby("year")["unit_price"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        baseline_yearly = test_baseline.groupby("year")["unit_price"].agg(
            ["mean", "std", "count"]
        ).reset_index()

        if len(strategy_yearly) >= 2:
            s_first = strategy_yearly.iloc[0]["mean"]
            s_last = strategy_yearly.iloc[-1]["mean"]
            strategy_return = (s_last - s_first) / s_first * 100
            years = max(len(strategy_yearly) - 1, 1)
            strategy_cagr = ((s_last / s_first) ** (1 / years) - 1) * 100
        else:
            strategy_return = np.nan
            strategy_cagr = np.nan

        if len(baseline_yearly) >= 2:
            b_first = baseline_yearly.iloc[0]["mean"]
            b_last = baseline_yearly.iloc[-1]["mean"]
            baseline_return = (b_last - b_first) / b_first * 100
            years_b = max(len(baseline_yearly) - 1, 1)
            baseline_cagr = ((b_last / b_first) ** (1 / years_b) - 1) * 100
        else:
            baseline_return = np.nan
            baseline_cagr = np.nan

        result.update({
            "strategy_return": strategy_return,
            "baseline_return": baseline_return,
            "strategy_cagr": strategy_cagr,
            "baseline_cagr": baseline_cagr,
            "strategy_volatility": strategy_yearly["std"].mean() if len(strategy_yearly) > 0 else np.nan,
            "baseline_volatility": baseline_yearly["std"].mean() if len(baseline_yearly) > 0 else np.nan,
            "strategy_test_count": len(test_strategy),
            "baseline_test_count": len(test_baseline),
        })

    # 预算适配性
    if "max_price" in strategy_config and strategy_config["max_price"]:
        if test_df is not None:
            eligible = len(test_strategy[test_strategy["resale_price"] <= strategy_config["max_price"]])
            result["budget_fit_rate"] = eligible / len(test_strategy) * 100 if len(test_strategy) > 0 else 0
        else:
            result["budget_fit_rate"] = 100  # 训练期策略组都符合预算

    return result


def analyze_event_impact(df, events_df, windows_months=[3, 6]):
    """
    分析政策事件对价格的影响

    Args:
        df: 包含 month 列和 unit_price 列的数据
        events_df: 事件数据
        windows_months: 前后对比的窗口期（月）

    Returns:
        pd.DataFrame: 事件影响分析结果
    """
    results = []
    df["month"] = pd.to_datetime(df["month"])

    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event["date"])
        event_name = event["event"]
        event_desc = event.get("description", "")

        for window in windows_months:
            before_start = event_date - pd.DateOffset(months=window)
            after_end = event_date + pd.DateOffset(months=window)

            before = df[(df["month"] >= before_start) & (df["month"] < event_date)]
            after = df[(df["month"] >= event_date) & (df["month"] < after_end)]

            if len(before) > 0 and len(after) > 0:
                before_avg = before["unit_price"].mean()
                after_avg = after["unit_price"].mean()
                change = (after_avg - before_avg) / before_avg * 100
            else:
                before_avg = np.nan
                after_avg = np.nan
                change = np.nan

            results.append({
                "事件": event_name,
                "描述": event_desc,
                "事件日期": event_date,
                "窗口(月)": window,
                "事件前均价": before_avg,
                "事件后均价": after_avg,
                "变化(%)": change,
                "事件前样本": len(before),
                "事件后样本": len(after),
            })

    return pd.DataFrame(results)


def compute_strategy_score(validation_result):
    """
    计算策略综合得分

    综合得分 = 策略收益得分 × 0.3 + 稳定性得分 × 0.2
             + 流动性得分 × 0.2 + 预算适配得分 × 0.1
             + 分析深度得分 × 0.2

    Args:
        validation_result: validate_strategy 的返回结果

    Returns:
        float: 综合得分（仅包含数据可计算的指标）
    """
    score = 0
    details = {}

    # 收益得分（策略组 vs 基准组年化涨幅差异）
    if "strategy_cagr" in validation_result and "baseline_cagr" in validation_result:
        s_cagr = validation_result["strategy_cagr"]
        b_cagr = validation_result["baseline_cagr"]
        if not np.isnan(s_cagr) and not np.isnan(b_cagr):
            diff = s_cagr - b_cagr
            # 超额收益越高越好，映射到0-10
            ret_score = min(max(diff * 2 + 5, 0), 10)
            score += ret_score * 0.3
            details["收益得分"] = ret_score

    # 稳定性得分（波动率越低越好）
    if "strategy_volatility" in validation_result:
        vol = validation_result["strategy_volatility"]
        if not np.isnan(vol):
            stab_score = max(10 - vol * 0.02, 0) if vol > 0 else 10
            score += stab_score * 0.2
            details["稳定性得分"] = stab_score

    # 流动性得分（成交量充足）
    if "strategy_test_count" in validation_result:
        count = validation_result["strategy_test_count"]
        liq_score = min(count / 100 * 10, 10) if count > 0 else 0
        score += liq_score * 0.2
        details["流动性得分"] = liq_score

    # 预算适配得分
    if "budget_fit_rate" in validation_result:
        fit_rate = validation_result["budget_fit_rate"]
        budget_score = min(fit_rate / 10, 10)
        score += budget_score * 0.1
        details["预算适配得分"] = budget_score

    # 分析深度得分由老师主观评定
    details["分析深度得分"] = "老师评定"
    details["综合得分"] = score

    return score, details
