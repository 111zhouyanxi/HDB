import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pydeck as pdk
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =====================================================
# 页面设置
# =====================================================

st.set_page_config(
    page_title="新加坡HDB组屋转售价格分析系统",
    layout="wide"
)

st.markdown("""
<style>

/* multiselect整体 */
.stMultiSelect [data-baseweb="select"] {
    background-color: white;
    border: 1px solid #dcdcdc;
    border-radius: 10px;
    min-height: 45px;
}

/* 已选择标签 */
.stMultiSelect span {
    background-color: #eef3f8 !important;
    color: #333333 !important;
    border-radius: 6px !important;
}

/* hover */
.stMultiSelect [data-baseweb="tag"]:hover {
    background-color: #dce7f3 !important;
}

</style>
""", unsafe_allow_html=True)

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 标题
# =====================================================

st.title("新加坡 HDB 组屋转售价格分析与预测系统")
st.markdown("---")

# =====================================================
# 读取数据
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("hdb_resale_data.csv")
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    df['house_age'] = 2026 - df['lease_commence_date']

    def extract_lease(x):
        match = re.search(r'(\d+)', str(x))
        if match:
            return int(match.group(1))
        return np.nan

    df['remaining_lease_year'] = df['remaining_lease'].apply(extract_lease)
    df['unit_price'] = df['resale_price'] / df['floor_area_sqm']
    return df

df = load_data()

# =====================================================
# 距离计算函数
# =====================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon1 - lon2)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# =====================================================
# 配套设施数据
# =====================================================

mrt_data = pd.read_csv("mrt_locations.csv")
school_data = pd.read_csv("school_locations.csv")

# =====================================================
# 镇区经纬度
# =====================================================

town_locations = pd.read_csv("town_locations.csv")
df = df.merge(town_locations, on="town", how="left")

# =====================================================
# MRT 距离计算
# =====================================================

df["mrt_distance"] = df.apply(
    lambda row: min([
        haversine(
            row["latitude"],
            row["longitude"],
            s["latitude"],
            s["longitude"]
        )
        for _, s in mrt_data.iterrows()
    ]),
    axis=1
)

# =====================================================
# 侧边栏
# =====================================================

st.sidebar.title("功能菜单")

menu = st.sidebar.radio(
    "请选择功能",
    [
        "数据概览",
        "地图可视化",
        "房价影响因素分析",
        "房价预测",
        "新加坡住房政策事件冲击分析"
    ]
)

# =====================================================
# 板块1数据概览
# =====================================================

if menu == "数据概览":
    st.header("HDB 数据筛选与统计分析")
    st.sidebar.header("数据筛选")

    towns = sorted(df["town"].unique())
    selected_towns = st.sidebar.multiselect("选择镇区", options=towns, placeholder="请选择镇区")
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    year_range = st.sidebar.slider("年份范围", min_year, max_year, (2020, max_year))

    flat_type = st.sidebar.selectbox("房型", ["不限", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])
    min_area = int(df["floor_area_sqm"].min())
    max_area = int(df["floor_area_sqm"].max())
    area_range = st.sidebar.slider("面积范围（㎡）", min_area, max_area, (60, 120))

    if len(selected_towns) == 0:
        selected_towns = towns
    filtered_df = df[
        (df["town"].isin(selected_towns)) &
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1]) &
        (df["floor_area_sqm"] >= area_range[0]) &
        (df["floor_area_sqm"] <= area_range[1])
    ]
    if flat_type != "不限":
        filtered_df = filtered_df[filtered_df["flat_type"] == flat_type]

    total_count = len(filtered_df)
    avg_price = filtered_df["unit_price"].mean()
    avg_total = filtered_df["resale_price"].mean()
    max_price = filtered_df["unit_price"].max()
    min_price = filtered_df["unit_price"].min()

    st.subheader("基础统计看板")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成交套数", f"{total_count} 套")
    col2.metric("平均单价", f"S${avg_price:.0f}/㎡")
    col3.metric("平均总价", f"S${avg_total:.0f}")
    col4.metric("最高单价", f"S${max_price:.0f}/㎡")
    st.metric("最低单价", f"S${min_price:.0f}/㎡")

    st.subheader("筛选后的成交记录")
    st.dataframe(filtered_df[["month", "town", "flat_type", "floor_area_sqm", "resale_price", "unit_price"]])

    st.subheader("镇区平均房价")
    if filtered_df.empty:
        st.warning("当前筛选条件下没有数据，请重新选择条件")
    else:
        town_avg = filtered_df.groupby("town")["resale_price"].mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        town_avg.plot(kind="bar", ax=ax)
        ax.set_title("各镇区平均房价")
        ax.set_ylabel("平均房价")
        st.pyplot(fig)

# =====================================================
# 模块2：地图可视化
# =====================================================

elif menu == "地图可视化":
    st.header("🗺️ HDB 房价地图可视化")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.select_slider("选择年份", options=sorted(df["year"].unique()))
    with col2:
        radius = st.slider("热力图半径", 20, 100, 60)
    with col3:
        show_layers = st.multiselect("叠加显示", ["MRT站点", "学校", "商场"], default=["MRT站点"])

    st.markdown("---")
    year_df = df[df["year"] == year]
    town_avg = year_df.groupby(["town", "latitude", "longitude"])["unit_price"].mean().reset_index()
    global_min = df["unit_price"].min()
    global_max = df["unit_price"].max()

    scatter_layer = pdk.Layer(
        "ScatterplotLayer", data=town_avg, get_position=["longitude", "latitude"],
        get_radius="unit_price * 0.4", get_fill_color=[255, 140, 0, 180], pickable=True, auto_highlight=True
    )
    heatmap_layer = pdk.Layer(
        "HeatmapLayer", data=town_avg, get_position=["longitude", "latitude"],
        get_weight="unit_price", radius_pixels=radius
    )

    mrt_layer = pdk.Layer(
        "ScatterplotLayer", data=mrt_data, get_position=["longitude", "latitude"],
        get_fill_color=[0, 128, 255], get_radius=120, pickable=True
    )
    school_layer = pdk.Layer(
        "ScatterplotLayer", data=school_data, get_position=["longitude", "latitude"],
        get_fill_color=[255, 0, 0], get_radius=120, pickable=True
    )
    mall_data = pd.read_csv("mall_locations.csv")
    mall_layer = pdk.Layer(
        "ScatterplotLayer", data=mall_data, get_position=["longitude", "latitude"],
        get_fill_color=[0, 200, 100], get_radius=120, pickable=True
    )

    layers = [heatmap_layer, scatter_layer]
    if "MRT站点" in show_layers:
        layers.append(mrt_layer)
    if "学校" in show_layers:
        layers.append(school_layer)
    if "商场" in show_layers:
        layers.append(mall_layer)

    view = pdk.ViewState(latitude=1.3521, longitude=103.8198, zoom=11, pitch=40)
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view, layers=layers,
        tooltip={
            "html": """<b>镇区:</b> {town}<br/><b>平均单价:</b> {unit_price} 元/㎡""",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    )
    st.pydeck_chart(deck, width='stretch')

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("当前年份", year)
    c2.metric("平均单价", f"S${town_avg['unit_price'].mean():.0f}")
    c3.metric("最高单价", f"S${town_avg['unit_price'].max():.0f}")

# =====================================================
# 模块3：房价影响因素分析（已修复：trendline=None）
# =====================================================

elif menu == "房价影响因素分析":
    st.header("📊 房价影响因素分析")
    st.markdown("---")
    current_year = 2026
    df["house_age"] = current_year - df["lease_commence_date"]

    def floor_category(x):
        if "01 TO 03" in x:
            return "低楼层"
        elif "04 TO 06" in x:
            return "中楼层"
        else:
            return "高楼层"

    df["floor_category"] = df["storey_range"].apply(floor_category)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["面积分析", "房型分析", "楼层分析", "租约分析", "房龄分析"])

    with tab1:
        st.subheader("面积 vs 单价")
        fig = px.scatter(df.sample(3000), x="floor_area_sqm", y="unit_price", trendline=None, title="面积与单价关系", opacity=0.5)
        st.plotly_chart(fig, width='stretch')

    with tab2:
        st.subheader("房型 vs 均价")
        flat_avg = df.groupby("flat_type")["resale_price"].mean().reset_index()
        fig = px.bar(flat_avg, x="flat_type", y="resale_price", title="不同房型均价对比")
        st.plotly_chart(fig, width='stretch')

    with tab3:
        st.subheader("楼层 vs 均价")
        fig = px.box(df, x="floor_category", y="unit_price", title="楼层与单价关系")
        st.plotly_chart(fig, width='stretch')

    with tab4:
        st.subheader("剩余租约 vs 单价")
        df["lease_years"] = df["remaining_lease"].str.extract(r"(\d+)").astype(float)
        fig = px.scatter(df.sample(3000), x="lease_years", y="unit_price", trendline=None, title="剩余租约与单价关系", opacity=0.5)
        st.plotly_chart(fig, width='stretch')

    with tab5:
        st.subheader("房龄 vs 单价")
        fig = px.scatter(df.sample(3000), x="house_age", y="unit_price", trendline=None, title="房龄与单价关系", opacity=0.5)
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("🏘️ 成熟区 vs 非成熟区分析")
    mature_towns = ["QUEENSTOWN", "TOA PAYOH", "ANG MO KIO", "BEDOK"]
    df["town_type"] = df["town"].apply(lambda x: "成熟区" if x in mature_towns else "非成熟区")

    with st.expander("查看成熟区定义"):
        st.write("""
        成熟组屋区：Queenstown、Toa Payoh、Ang Mo Kio、Bedok 等。
        非成熟区：Punggol、Sengkang 等新兴发展区域。
        """)

    town_compare = df.groupby("town_type")["resale_price"].mean().reset_index()
    fig = px.bar(town_compare, x="town_type", y="resale_price", color="town_type", title="成熟区 vs 非成熟区均价")
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader("🚇 MRT距离对房价影响")
    fig = px.scatter(df.sample(3000), x="mrt_distance", y="unit_price", trendline=None, title="MRT距离 vs 单价", opacity=0.5)
    st.plotly_chart(fig, width='stretch')

    near_mrt = df[df["mrt_distance"] < 0.5]["unit_price"].mean()
    far_mrt = df[df["mrt_distance"] > 1]["unit_price"].mean()
    c1, c2 = st.columns(2)
    c1.metric("距离 MRT < 500m", f"S${near_mrt:.0f}/㎡")
    c2.metric("距离 MRT > 1km", f"S${far_mrt:.0f}/㎡")

    st.markdown("---")
    st.subheader("📈 购房策略与保值分析")
    st.write("策略：选择成熟区 + MRT 500m内 + 4 ROOM + 总价低于 600000 新币。")

    strategy_df = df[(df["town_type"] == "成熟区") & (df["mrt_distance"] < 0.5) & (df["flat_type"] == "4 ROOM") & (df["resale_price"] < 600000)]
    baseline_df = df[(df["flat_type"] == "4 ROOM")]

    c1, c2, c3 = st.columns(3)
    c1.metric("策略组均价", f"S${strategy_df['resale_price'].mean():.0f}")
    c2.metric("基准组均价", f"S${baseline_df['resale_price'].mean():.0f}")
    c3.metric("策略组成交量", len(strategy_df))

    strategy_trend = strategy_df.groupby("year")["resale_price"].mean().reset_index()
    baseline_trend = baseline_df.groupby("year")["resale_price"].mean().reset_index()

    fig = px.line(title="策略组 vs 基准组价格走势")
    fig.add_scatter(x=strategy_trend["year"], y=strategy_trend["resale_price"], mode="lines+markers", name="策略组")
    fig.add_scatter(x=baseline_trend["year"], y=baseline_trend["resale_price"], mode="lines+markers", name="基准组")
    st.plotly_chart(fig, width='stretch')

# =====================================================
# 其他页面（空页面不报错）
# =====================================================

elif menu == "房价预测":
    st.header("房价预测")
    st.info("预测模块可在此扩展")

elif menu == "新加坡住房政策事件冲击分析":
    st.header("政策冲击分析")
    st.info("政策模块可在此扩展")
