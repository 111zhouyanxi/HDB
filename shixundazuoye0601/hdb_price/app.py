"""
新加坡 HDB 组屋转售价格分析与预测系统
Streamlit 主程序

项目结构：
- app.py          : Streamlit 主程序
- data/           : 数据文件目录
- utils/          : 工具模块目录
  - data_loader.py : 数据加载与清洗
  - geo_utils.py   : 地理工具函数
  - feature_eng.py : 特征工程
  - model.py       : 模型训练与预测（已在文件内部全字段去重，修复narwhals报错）
  - analysis.py    : 分析函数
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path
import sys

# ========== 页面配置 ==========
st.set_page_config(
    page_title="Singapore HDB Resale Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 全局 plotly 配色 — 禁用红色，统一轻奢金暖调
LUX_COLORS = ["#c9a96e", "#8b9a8b", "#b8a88c", "#7b8c7b", "#d4c5a9", "#a89880",
              "#b8956a", "#6b8b6b", "#c4b5a0", "#9b927e"]
px.defaults.color_discrete_sequence = LUX_COLORS
px.defaults.color_continuous_scale = ["#d4c5a9", "#f0e8d5", "#c9a96e", "#b8956a", "#8b6914"]

# ========== 轻奢风 CSS ==========
st.markdown("""
<style>
/* ==== Font & Global ==== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, 'PingFang SC', sans-serif; color: #3d3929; }

.stApp { background: #faf9f6; }

/* ==== Sidebar ==== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f1ea 0%, #faf9f6 100%);
    border-right: 1px solid #e8e3d9;
}
[data-testid="stSidebar"] * { color: #5c5544 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #3d3929 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    color: #9b927e !important; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500;
}

/* ==== Headings ==== */
h1 { font-family: 'DM Serif Display', 'Georgia', serif !important; font-size: 2rem !important;
     font-weight: 400 !important; color: #2d2920 !important; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem !important; font-weight: 500 !important; color: #5c5544 !important;
     border-bottom: 1px solid #e8e3d9; padding-bottom: 0.4rem; }
h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #9b927e !important; }

/* ==== Metric cards ==== */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ede8df;
    border-radius: 12px;
    padding: 1.2rem 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #d4b896;
    box-shadow: 0 4px 16px rgba(180,150,110,0.12);
}
[data-testid="stMetric"] label {
    font-size: 0.68rem !important; text-transform: uppercase;
    letter-spacing: 0.1em; color: #b8a88c !important; font-weight: 500;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', monospace !important; font-size: 1.55rem !important;
    font-weight: 600 !important; color: #2d2920 !important;
}

/* ==== DataFrame ==== */
[data-testid="stDataFrame"] { border: 1px solid #ede8df; border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] th {
    background: #faf8f3 !important; color: #9b927e !important;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}
[data-testid="stDataFrame"] td { font-size: 0.82rem; color: #5c5544; }

/* ==== Tabs ==== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: #ffffff; border-radius: 12px; border: 1px solid #ede8df; padding: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; padding: 0.5rem 1.3rem !important;
    font-size: 0.82rem !important; color: #9b927e !important; font-weight: 500;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #c9a96e, #d4b896) !important;
    color: #fff !important; box-shadow: 0 2px 8px rgba(180,140,100,0.25);
}

/* ==== Expander ==== */
[data-testid="stExpander"] {
    border: 1px solid #ede8df !important; border-radius: 12px !important;
    background: #ffffff !important; box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

/* ==== Buttons ==== */
.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #b8956a) !important;
    border: none !important; color: #fff !important; border-radius: 8px !important;
    font-weight: 500 !important; letter-spacing: 0.02em; transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(180,140,100,0.2);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #d4b896, #c9a96e) !important;
    box-shadow: 0 4px 16px rgba(180,140,100,0.3); transform: translateY(-1px);
}

/* ==== Alert boxes ==== */
[data-testid="stAlert"] { border-radius: 10px; font-size: 0.85rem; border: none; }
div[data-testid="stAlert-info"] { background: #fcf9f2; border-left: 3px solid #d4b896; }

/* ==== blockquote ==== */
blockquote {
    border-left: 3px solid #d4b896 !important; color: #9b927e !important;
    padding: 0.5rem 1rem; margin: 0.5rem 0; font-style: italic; background: #fdfcf9;
    border-radius: 0 6px 6px 0;
}

/* ==== Caption text ==== */
.stCaption { color: #b8a88c !important; font-size: 0.8rem; }

/* ==== Select / Slider ==== */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color: #5c5544 !important; }

/* ==== Radio ==== */
[data-testid="stRadio"] label { color: #5c5544 !important; }

/* ==== Divider ==== */
hr { border-color: #ede8df !important; }
</style>
""", unsafe_allow_html=True)

# ========== 导入工具模块 ==========
# 将项目根目录加入 Python 路径
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.data_loader import (
    load_hdb_data, load_town_locations, load_mrt_stations,
    load_schools, load_events, get_filtered_data, get_stats, get_data_dir,
)
from utils.geo_utils import (
    haversine, find_nearest_distance, merge_town_coords,
    compute_town_distances, classify_mrt_proximity,
)
from utils.feature_eng import (
    build_features, get_feature_columns, MATURE_ESTATES,
    classify_flat_category, BASE_FEATURES,
)
from utils.model import (
    train_model, evaluate_model, evaluate_by_category,
    get_feature_importance, predict_price, find_prediction_errors,
    prepare_model_data, MODEL_CONFIGS,
    cross_validate_model, train_all_models,
)
from utils.analysis import (
    analyze_town_price_trend, analyze_preservation_value,
    validate_strategy, analyze_event_impact, compute_strategy_score,
)


# ========== 数据加载（缓存） ==========
@st.cache_data(ttl=3600)
def load_all_data():
    """加载所有数据文件"""
    data_dir = get_data_dir()
    hdb_path = data_dir / "hdb_resale.csv"

    # 检查主数据文件
    if not hdb_path.exists():
        # 尝试从桌面实训大作业目录加载原始数据
        alt_path = Path.home() / "Desktop" / "实训大作业" / "ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv"
        if alt_path.exists():
            st.info(f"从 {alt_path} 加载原始数据...")
            df = load_hdb_data(str(alt_path))
            # 过滤 2020 年至今
            df = df[df["year"] >= 2020].reset_index(drop=True)
        else:
            st.error("未找到 HDB 转售数据文件！请将 hdb_resale.csv 放入 data/ 目录。")
            st.stop()
    else:
        df = load_hdb_data(str(hdb_path))

    town_locations = load_town_locations()
    mrt_stations = load_mrt_stations()
    schools = load_schools()
    events = load_events()

    # 合并镇区坐标和 estate_type 到成交数据中
    df = merge_town_coords(df, town_locations)

    # 计算镇区级到最近 MRT 和学校的距离
    # 为每条记录计算到最近 MRT 的距离
    if not mrt_stations.empty and "latitude" in df.columns:
        town_centers = df[["town", "latitude", "longitude"]].drop_duplicates()
        distances_mrt = {}
        distances_school = {}
        for _, row in town_centers.iterrows():
            t = row["town"]
            lat, lon = row["latitude"], row["longitude"]
            if pd.notna(lat) and pd.notna(lon):
                distances_mrt[t] = find_nearest_distance(lat, lon, mrt_stations)
                distances_school[t] = find_nearest_distance(lat, lon, schools)
        df["mrt_distance"] = df["town"].map(distances_mrt)
        df["school_distance"] = df["town"].map(distances_school)

    return df, town_locations, mrt_stations, schools, events


# ========== 加载数据 ==========
with st.spinner("正在加载数据..."):
    df, town_locations, mrt_stations, schools, events = load_all_data()

# ========== 侧边栏 ==========
st.sidebar.title("🏠 HDB 价格分析系统")
st.sidebar.markdown("---")

# 全局筛选
st.sidebar.header("🔍 数据筛选")

# 镇区选择
all_towns = sorted(df["town"].unique().tolist())
default_towns = ["TAMPINES", "BEDOK"]
default_towns = [t for t in default_towns if t in all_towns]
selected_towns = st.sidebar.multiselect(
    "选择镇区",
    options=all_towns,
    default=default_towns,
)

# 年份范围
min_year, max_year = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "年份范围",
    min_value=min_year,
    max_value=max_year,
    value=(2020, max_year),
)

# 房型筛选
flat_types = ["不限"] + sorted(df["flat_type"].unique().tolist())
selected_flat_type = st.sidebar.selectbox("房型", options=flat_types)

# 面积筛选
area_min, area_max = int(df["floor_area_sqm"].min()), int(df["floor_area_sqm"].max())
area_range = st.sidebar.slider(
    "面积范围（㎡）",
    min_value=area_min,
    max_value=area_max,
    value=(60, 120),
)

st.sidebar.markdown("---")

# 地图叠加选项（全局使用）
st.sidebar.header("🗺️ 地图叠加")
show_mrt = st.sidebar.checkbox("显示 MRT 站点", value=True)
show_schools = st.sidebar.checkbox("显示学校", value=False)

# 模型参数（全局使用）
st.sidebar.header("🤖 模型参数")
selected_model_name = st.sidebar.selectbox(
    "选择模型",
    options=list(MODEL_CONFIGS.keys()),
    format_func=lambda x: MODEL_CONFIGS[x]["label"],
)

if selected_model_name == "RandomForest":
    n_estimators = st.sidebar.slider("树的数量", 50, 300, 100, 50)
    max_depth = st.sidebar.slider("最大深度", 3, 30, 10)
    model_kwargs = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
elif selected_model_name == "GradientBoosting":
    n_estimators = st.sidebar.slider("迭代次数", 50, 300, 100, 50)
    max_depth = st.sidebar.slider("最大深度", 2, 15, 5)
    lr = st.sidebar.slider("学习率", 0.01, 0.5, 0.1, 0.01)
    model_kwargs = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": lr, "random_state": 42}
elif selected_model_name == "Ridge":
    alpha = st.sidebar.slider("正则化强度 (alpha)", 0.01, 10.0, 1.0, 0.01)
    model_kwargs = {"alpha": alpha}
else:
    model_kwargs = {}

st.sidebar.markdown("---")
st.sidebar.caption("实训大作业 · 新加坡HDB组屋转售价格分析")
st.sidebar.caption("数据来源: data.gov.sg")

# ========== 筛选数据 ==========
filtered_df = get_filtered_data(
    df,
    towns=selected_towns,
    year_range=year_range,
    flat_type=selected_flat_type,
    area_range=area_range,
)

# ========== 主页面 Tabs ==========
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 数据总览",
    "🗺️ 地图可视化",
    "📈 影响因素分析",
    "🤖 价格预测",
    "💡 分析思考",
    "🏆 策略验证",
    "📰 政策事件分析（可选）",
])

# ================================================================
# Tab 1: 数据总览
# ================================================================
with tab1:
    st.title("📊 数据总览")
    st.markdown("> 展示筛选后的 HDB 转售成交数据及基础统计信息")

    # 统计看板
    stats = get_stats(filtered_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("成交套数", f"{stats['count']} 套")
    with col2:
        st.metric("平均单价", f"S${stats['avg_unit_price']:,.0f}/㎡")
    with col3:
        st.metric("平均总价", f"S${stats['avg_total_price']:,.0f}")
    with col4:
        st.metric("最高单价", f"S${stats['max_unit_price']:,.0f}/㎡")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("最低单价", f"S${stats['min_unit_price']:,.0f}/㎡")
    with col2:
        st.metric("单价中位数", f"S${stats['median_unit_price']:,.0f}/㎡")
    with col3:
        st.metric("平均面积", f"{stats['avg_area']:.1f} ㎡")
    with col4:
        st.metric("平均剩余租约", f"{stats['avg_remaining_years']:.1f} 年")

    st.markdown("---")

    # 数据表格
    st.subheader("📋 筛选后的成交记录")

    display_cols = [
        "month", "town", "flat_type", "street_name", "storey_range",
        "floor_area_sqm", "flat_model", "lease_commence_date",
        "remaining_lease", "resale_price", "unit_price"
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    # 格式化显示
    display_df = filtered_df[available_cols].copy()
    if "month" in display_df.columns:
        display_df["month"] = display_df["month"].astype(str)
    if "unit_price" in display_df.columns:
        display_df["unit_price"] = display_df["unit_price"].round(0)
    if "resale_price" in display_df.columns:
        display_df["resale_price"] = display_df["resale_price"].round(0)

    st.dataframe(display_df, use_container_width=True, height=400)

    # 数据概览
    with st.expander("📊 数据分布概览"):
        col1, col2 = st.columns(2)
        with col1:
            # 各镇区成交量
            town_counts = filtered_df["town"].value_counts().reset_index()
            town_counts.columns = ["镇区", "成交量"]
            fig = px.bar(town_counts, x="镇区", y="成交量",
                         title="各镇区成交量分布", color="成交量")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 各房型分布
            flat_counts = filtered_df["flat_type"].value_counts().reset_index()
            flat_counts.columns = ["房型", "数量"]
            fig = px.pie(flat_counts, values="数量", names="房型",
                         title="房型分布")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📈 年度成交趋势"):
        yearly_counts = filtered_df.groupby("year").size().reset_index(name="成交量")
        yearly_avg = filtered_df.groupby("year")["unit_price"].mean().reset_index(name="均价")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=yearly_counts["year"], y=yearly_counts["成交量"],
                             name="成交量", yaxis="y", marker_color="lightblue"))
        fig.add_trace(go.Scatter(x=yearly_avg["year"], y=yearly_avg["均价"],
                                 name="均价 (S$/㎡)", yaxis="y2",
                                 mode="lines+markers", line=dict(color="#c9a96e", width=2)))
        fig.update_layout(
            title="年度成交量与均价趋势",
            xaxis_title="年份",
            yaxis=dict(title="成交量（套）"),
            yaxis2=dict(title="均价（S$/㎡）", overlaying="y", side="right"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# Tab 2: 地图可视化
# ================================================================
with tab2:
    st.title("🗺️ 地图可视化")
    st.markdown("> 展示镇区均价地图、房价热力图、时空变化及配套设施叠加")

    # 年份选择器
    map_year = st.select_slider(
        "选择年份",
        options=list(range(min_year, max_year + 1)),
        value=max_year,
    )

    # 配套叠加选择
    overlay_options = []
    if show_mrt:
        overlay_options.append("MRT站点")
    if show_schools:
        overlay_options.append("学校")

    # 准备地图数据：按镇区聚合
    year_data = filtered_df[filtered_df["year"] == map_year]
    if len(year_data) == 0:
        st.warning(f"{map_year} 年暂无筛选数据，请调整筛选条件。")
    else:
        town_agg = year_data.groupby("town").agg(
            avg_unit_price=("unit_price", "mean"),
            avg_total_price=("resale_price", "mean"),
            transaction_count=("resale_price", "count"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        ).reset_index()
        town_agg = town_agg.dropna(subset=["latitude", "longitude"])

        # ---- 地图模式选择 ----
        map_mode = st.radio(
            "地图显示模式",
            options=["镇区均价地图", "热力地图", "Hexagon 聚合"],
            horizontal=True,
        )

        layers = []

        if map_mode == "镇区均价地图":
            # 香槟金→深金色渐变（轻奢风）
            price_min = town_agg["avg_unit_price"].min()
            price_max = town_agg["avg_unit_price"].max()
            price_range = max(price_max - price_min, 1)
            ratio = (town_agg["avg_unit_price"] - price_min) / price_range
            town_agg["color_r"] = (212 - 28 * ratio).astype(int)
            town_agg["color_g"] = (175 - 70 * ratio).astype(int)
            town_agg["color_b"] = (120 - 95 * ratio).astype(int)

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=town_agg,
                get_position=["longitude", "latitude"],
                get_fill_color="[color_r, color_g, color_b, 200]",
                get_radius="avg_unit_price * 1.5",
                radius_min_pixels=15,
                radius_max_pixels=60,
                pickable=True,
                opacity=0.7,
            )
            layers.append(scatter_layer)

        elif map_mode == "热力地图":
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=town_agg,
                get_position=["longitude", "latitude"],
                get_weight="avg_unit_price",
                radius_pixels=60,
                intensity=0.5,
                threshold=0.1,
            )
            layers.append(heatmap_layer)

        elif map_mode == "Hexagon 聚合":
            hex_layer = pdk.Layer(
                "HexagonLayer",
                data=town_agg,
                get_position=["longitude", "latitude"],
                get_elevation_weight="avg_unit_price",
                elevation_scale=50,
                radius=3000,
                elevation_range=[0, 1000],
                extruded=True,
                pickable=True,
                coverage=0.9,
            )
            layers.append(hex_layer)

        # MRT 站点叠加层
        if show_mrt and not mrt_stations.empty:
            mrt_layer = pdk.Layer(
                "ScatterplotLayer",
                data=mrt_stations,
                get_position=["longitude", "latitude"],
                get_fill_color="[100, 160, 150, 200]",
                get_radius=100,
                radius_min_pixels=4,
                radius_max_pixels=8,
                pickable=True,
                opacity=0.8,
                stroked=True,
            )
            layers.append(mrt_layer)

        # 学校叠加层
        if show_schools and not schools.empty:
            school_data = schools.dropna(subset=["latitude", "longitude"])
            school_layer = pdk.Layer(
                "ScatterplotLayer",
                data=school_data,
                get_position=["longitude", "latitude"],
                get_fill_color="[210, 180, 140, 200]",
                get_radius=80,
                radius_min_pixels=3,
                radius_max_pixels=7,
                pickable=True,
                opacity=0.8,
            )
            layers.append(school_layer)

        # 视图配置
        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=10.5,
            pitch=40 if map_mode != "Hexagon 聚合" else 60,
        )

        # 图例
        st.caption(f"📍 {map_year} 年 | 显示 {len(town_agg)} 个镇区")

        # 渲染地图（Carto 免费底图，无需 token）
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{town}\n均价: S${avg_unit_price:.0f}/㎡\n成交量: {transaction_count} 套"},
            map_style="light",
        ))

        # 备用：如果 pydeck 不显示，用 st.map 兜底
        with st.expander("📌 备用简易地图（如上方地图不显示请展开）"):
            map_df = town_agg.rename(columns={"latitude": "lat", "longitude": "lon"})
            st.map(map_df[["lat", "lon", "avg_unit_price"]].dropna(),
                   size="avg_unit_price", color=None)

        # 图例说明
        col1, col2, col3 = st.columns(3)
        with col1:
            if show_mrt:
                st.markdown("🟢 **绿色点**: MRT 站点")
        with col2:
            if show_schools:
                st.markdown("🟠 **橙色点**: 学校")
        with col3:
            if map_mode == "镇区均价地图":
                st.markdown("🔵 **蓝色圆**: 镇区均价（越大/越紫 = 价格越高）")

    # ---- 时空变化分析 ----
    st.markdown("---")
    st.subheader("📅 各镇区年度均价变化")

    if len(selected_towns) > 0:
        yearly_town = filtered_df.groupby(["town", "year"])["unit_price"].mean().reset_index()
        yearly_town = yearly_town[yearly_town["town"].isin(selected_towns)]

        fig = px.line(
            yearly_town,
            x="year", y="unit_price", color="town",
            title="各镇区年度均价走势",
            labels={"unit_price": "均价 (S$/㎡)", "year": "年份", "town": "镇区"},
            markers=True,
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("请在侧边栏选择至少一个镇区以查看走势图。")


# ================================================================
# Tab 3: 影响因素分析
# ================================================================
with tab3:
    st.title("📈 房价影响因素分析")
    st.markdown("> 分析影响 HDB 组屋价格的各项因素")

    # 定义分析所用的数据
    analysis_df = filtered_df.copy()

    # ---- 因素1：面积 vs 单价 ----
    st.subheader("1️⃣ 建筑面积 vs 单价")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            analysis_df,
            x="floor_area_sqm", y="unit_price",
            color="flat_type",

            title="建筑面积与单价关系（按房型着色）",
            labels={
                "floor_area_sqm": "建筑面积（㎡）",
                "unit_price": "单价（S$/㎡）",
                "flat_type": "房型",
            },
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**观察要点**")
        st.markdown("- 大户型单价通常低于小户型")
        st.markdown("- 同面积不同房龄价格差异大")
        st.markdown("- 关注异常离群点（极低单价）")

    # ---- 因素2：房型 vs 均价 ----
    st.subheader("2️⃣ 房型 vs 均价")
    col1, col2 = st.columns(2)
    with col1:
        flat_stats = analysis_df.groupby("flat_type").agg(
            均价=("unit_price", "mean"),
            数量=("resale_price", "count"),
            标准差=("unit_price", "std"),
        ).reset_index()
        fig = px.bar(
            flat_stats, x="flat_type", y="均价",
            title="各房型均价对比",
            color="均价",
            text=flat_stats["均价"].apply(lambda x: f"S${x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(
            flat_stats.style.format({
                "均价": "S${:,.0f}",
                "标准差": "S${:,.0f}",
            }),
            use_container_width=True,
        )

    # ---- 因素3：楼层 vs 均价 ----
    st.subheader("3️⃣ 楼层 vs 均价")
    if "storey_mid" in analysis_df.columns:
        analysis_df["楼层类别"] = pd.cut(
            analysis_df["storey_mid"],
            bins=[0, 6, 15, 100],
            labels=["低楼层 (1-6)", "中楼层 (7-15)", "高楼层 (16+)"],
        )
        fig = px.box(
            analysis_df.dropna(subset=["楼层类别"]),
            x="楼层类别", y="unit_price",
            color="楼层类别",
            title="不同楼层类别的单价分布",
            labels={"unit_price": "单价 (S$/㎡)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("楼层数据不可用")

    # ---- 因素4：剩余租约 vs 单价 ----
    st.subheader("4️⃣ 剩余租约 vs 单价")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(
            analysis_df,
            x="remaining_years", y="unit_price",
            color="flat_type",

            title="剩余租约年限与单价关系",
            labels={
                "remaining_years": "剩余租约（年）",
                "unit_price": "单价（S$/㎡）",
                "flat_type": "房型",
            },
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**观察要点**")
        st.markdown("- 租约越短，单价越低")
        st.markdown("- 剩余<60年的组屋贬值加速")
        st.markdown("- 新租约（>90年）溢价明显")

    # ---- 因素5：房龄 vs 单价 ----
    st.subheader("5️⃣ 房龄 vs 单价")
    fig = px.scatter(
        analysis_df,
        x="flat_age", y="unit_price",
        color="is_mature" if "is_mature" in analysis_df.columns else None,
        title="房龄与单价关系",
        labels={
            "flat_age": "房龄（年）",
            "unit_price": "单价（S$/㎡）",
        },
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 因素6：成熟区 vs 非成熟区对比 ----
    st.markdown("---")
    st.subheader("6️⃣ 成熟区 vs 非成熟区对比分析")

    with st.expander("📖 成熟区/非成熟区定义说明"):
        st.markdown("""
        **成熟组屋区（Mature Estate）**：开发历史悠久、基础设施完善、交通便利、配套成熟的镇区。
        代表：Queenstown、Toa Payoh、Ang Mo Kio、Bedok、Tampines、Clementi 等。

        **非成熟组屋区（Non-mature Estate）**：发展较晚、仍有大量新组屋供应的新兴镇区。
        代表：Punggol、Sengkang、Woodlands、Jurong West、Bukit Batok 等。

        > 成熟区与非成熟区的价格差异，是新加坡组屋市场中最典型的结构性特征。
        """)

    col1, col2 = st.columns(2)
    with col1:
        # 成熟区 vs 非成熟区均价对比
        analysis_df["estate_type_label"] = analysis_df["is_mature"].apply(
            lambda x: "成熟区" if x == 1 else "非成熟区"
        ) if "is_mature" in analysis_df.columns else "未知"

        estate_stats = analysis_df.groupby("estate_type_label").agg(
            均价=("unit_price", "mean"),
            总价=("resale_price", "mean"),
            成交量=("resale_price", "count"),
            平均租约=("remaining_years", "mean"),
        ).reset_index()

        fig = px.bar(
            estate_stats, x="estate_type_label", y="均价",
            title="成熟区 vs 非成熟区均价对比",
            color="estate_type_label",
            text=estate_stats["均价"].apply(lambda x: f"S${x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 成熟区 vs 非成熟区价格走势
        estate_trend = analysis_df.groupby(["year", "estate_type_label"])["unit_price"].mean().reset_index()
        fig = px.line(
            estate_trend,
            x="year", y="unit_price", color="estate_type_label",
            title="成熟区 vs 非成熟区年度均价走势",
            labels={"unit_price": "均价 (S$/㎡)", "year": "年份"},
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        estate_stats.style.format({
            "均价": "S${:,.0f}",
            "总价": "S${:,.0f}",
            "平均租约": "{:.1f}年",
        }),
        use_container_width=True,
    )

    # ---- 因素7：MRT 距离 vs 单价 ----
    st.markdown("---")
    st.subheader("7️⃣ 配套设施对价格的影响")

    if "mrt_distance" in analysis_df.columns:
        analysis_df["mrt_category"] = analysis_df["mrt_distance"].apply(
            classify_mrt_proximity
        )

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                analysis_df,
                x="mrt_distance", y="unit_price",
                title="MRT 距离 vs 单价",
                labels={
                    "mrt_distance": "到最近 MRT 距离（km）",
                    "unit_price": "单价（S$/㎡）",
                },
    
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            mrt_stats = analysis_df.groupby("mrt_category")["unit_price"].agg(
                ["mean", "std", "count"]
            ).reset_index()
            mrt_stats.columns = ["MRT邻近度", "均价", "标准差", "成交量"]
            fig = px.bar(
                mrt_stats, x="MRT邻近度", y="均价",
                title="MRT 距离对价格的影响",
                color="MRT邻近度",
                text=mrt_stats["均价"].apply(lambda x: f"S${x:,.0f}"),
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    # 名校圈分析
    if "school_distance" in analysis_df.columns:
        st.subheader("🏫 学校距离对价格的影响")
        analysis_df["school_category"] = pd.cut(
            analysis_df["school_distance"],
            bins=[0, 0.5, 1.0, 100],
            labels=["< 500m", "500m - 1km", "> 1km"],
        )
        school_stats = analysis_df.groupby("school_category")["unit_price"].mean().reset_index()
        fig = px.bar(
            school_stats, x="school_category", y="unit_price",
            title="到学校距离 vs 均价",
            color="unit_price",
            text=school_stats["unit_price"].apply(lambda x: f"S${x:,.0f}"),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# Tab 4: 价格预测
# ================================================================
with tab4:
    st.title("🤖 房价预测模型")
    st.markdown("> 使用机器学习模型预测 HDB 组屋单价，对比分房型预测效果")

    # ---- 准备工作 ----
    with st.spinner("正在进行特征工程和模型训练..."):
        all_features_df = build_features(df, town_locations, mrt_stations, schools)
        feature_cols = get_feature_columns(all_features_df, include_town_dummies=True)
        train_df, test_df = prepare_model_data(all_features_df)

        # 训练模型 + 交叉验证 (log 变换目标)
        model, X_train, y_train_raw, used_features, log_used = train_model(
            all_features_df[all_features_df["year"] <= 2023],
            feature_cols,
            model_name=selected_model_name,
            use_log_target=True,
            **model_kwargs,
        )

        cv_results = cross_validate_model(
            all_features_df, feature_cols,
            model_name=selected_model_name, n_splits=3,
            use_log_target=True, **model_kwargs,
        )

        # 评估模型（测试集）
        X_test_df = test_df[used_features].dropna()
        test_idx = X_test_df.index
        y_test = test_df.loc[test_idx, "unit_price"]
        eval_results = evaluate_model(model, X_test_df, y_test, use_log_target=log_used)

        # 检查配置
        config = MODEL_CONFIGS[selected_model_name]
        using_scaler = config["need_scaler"]

    # ---- 模型配置提示 ----
    if using_scaler:
        st.info(f"🔧 **{config['label']}** 已自动启用 **StandardScaler** 特征标准化（线性模型需要统一量纲）")
    else:
        st.info(f"🌲 **{config['label']}** 为树模型，无需特征标准化")

    # ---- 整体评估指标 ----
    st.subheader("📊 测试集评估（2024+）")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"S${eval_results['MAE']:,.0f}/㎡" if not np.isnan(eval_results['MAE']) else "N/A",
                  help="平均绝对误差，越小越好")
    with col2:
        st.metric("RMSE", f"S${eval_results['RMSE']:,.0f}/㎡" if not np.isnan(eval_results['RMSE']) else "N/A",
                  help="均方根误差，对大误差惩罚更大")
    with col3:
        st.metric("R²", f"{eval_results['R2']:.3f}" if not np.isnan(eval_results['R2']) else "N/A",
                  help="决定系数，越接近1越好")
    with col4:
        st.metric("MAPE", f"{eval_results['MAPE']:.1f}%" if not np.isnan(eval_results.get('MAPE', np.nan)) else "N/A",
                  help="平均百分比误差")

    # ---- 交叉验证结果 ----
    st.subheader("🔄 时序交叉验证（TimeSeriesSplit）")
    st.caption("按年份顺序切分 3 折，考察模型在不同时间段上的稳定性")

    if len(cv_results["fold_details"]) > 0:
        cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
        with cv_col1:
            st.metric("CV-R² 均值", f"{cv_results['mean_r2']:.4f}" if not np.isnan(cv_results['mean_r2']) else "N/A")
        with cv_col2:
            st.metric("CV-R² 标准差", f"{cv_results['std_r2']:.4f}" if not np.isnan(cv_results['std_r2']) else "N/A",
                      help="越小说明模型越稳定")
        with cv_col3:
            st.metric("CV-MAE 均值", f"S${cv_results['mean_mae']:,.0f}" if not np.isnan(cv_results['mean_mae']) else "N/A")
        with cv_col4:
            st.metric("CV-MAE 标准差", f"S${cv_results['std_mae']:,.0f}" if not np.isnan(cv_results['std_mae']) else "N/A")

        # 折详情
        with st.expander("📋 各折详情"):
            st.dataframe(pd.DataFrame(cv_results["fold_details"]), use_container_width=True)
    else:
        st.warning("数据不足以进行交叉验证（年份太少）")

    # ---- 预测值 vs 实际值散点图 ----
    st.subheader("📈 预测值 vs 实际值")
    if "y_test" in eval_results and "y_pred" in eval_results:
        pred_df = pd.DataFrame({
            "实际单价": eval_results["y_test"].values,
            "预测单价": eval_results["y_pred"],
        })

        # 添加房型信息
        common_idx = list(set(eval_results["y_test"].index) & set(test_df.index))
        if common_idx:
            pred_df["房型"] = test_df.loc[eval_results["y_test"].index, "flat_type"].values

        fig = px.scatter(
            pred_df,
            x="实际单价", y="预测单价",
            color="房型" if "房型" in pred_df.columns else None,
            title="预测值 vs 实际值散点图",
            labels={"实际单价": "实际单价 (S$/㎡)", "预测单价": "预测单价 (S$/㎡)"},
            opacity=0.6,
        )
        # 添加完美预测线
        max_val = max(pred_df["实际单价"].max(), pred_df["预测单价"].max())
        min_val = min(pred_df["实际单价"].min(), pred_df["预测单价"].min())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="完美预测",
            line=dict(color="#9b927e", dash="dash", width=1),
        ))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 分房型预测对比 ----
    st.markdown("---")
    st.subheader("🏘️ 分类型预测效果对比")

    # 添加分类列到测试集
    if "flat_type" in test_df.columns:
        # 大小户型分类
        cat_df = test_df.copy()
        cat_df["size_cat"] = cat_df["flat_type"].apply(
            lambda x: "小户型" if x in ["2 ROOM", "3 ROOM"]
            else ("中户型" if x == "4 ROOM" else "大户型")
        )

        # 新旧分类
        cat_df["age_cat"] = cat_df["remaining_years"].apply(
            lambda x: "老旧组屋" if x < 60 else ("新近组屋" if x >= 80 else "中年组屋")
        )

        # MRT 分类
        if "mrt_distance" in cat_df.columns:
            cat_df["mrt_cat"] = cat_df["mrt_distance"].apply(
                lambda x: "MRT沿线" if x <= 0.5 else "远离MRT"
            )

        # 按户型分类评估
        size_labels = {"小户型": "小户型", "中户型": "中户型", "大户型": "大户型"}
        size_results = evaluate_by_category(cat_df, used_features, model, "size_cat", size_labels, use_log_target=log_used)

        # 按新旧分类评估
        age_labels = {"老旧组屋": "老旧组屋", "新近组屋": "新近组屋"}
        age_results = evaluate_by_category(cat_df, used_features, model, "age_cat", age_labels, use_log_target=log_used)

        # 按 MRT 分类评估
        if "mrt_cat" in cat_df.columns:
            mrt_labels = {"MRT沿线": "MRT沿线", "远离MRT": "远离MRT"}
            mrt_results = evaluate_by_category(cat_df, used_features, model, "mrt_cat", mrt_labels, use_log_target=log_used)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**按户型分类**")
            st.dataframe(
                size_results.style.format({
                    "MAE": "S${:,.0f}",
                    "MAPE": "{:.1f}%",
                    "R²": "{:.3f}",
                }),
                use_container_width=True,
            )

            st.markdown("**按新旧分类**")
            st.dataframe(
                age_results.style.format({
                    "MAE": "S${:,.0f}",
                    "R²": "{:.3f}",
                }),
                use_container_width=True,
            )

        with col2:
            st.markdown("**按 MRT 距离分类**")
            if "mrt_cat" in cat_df.columns:
                st.dataframe(
                    mrt_results.style.format({
                        "MAE": "S${:,.0f}",
                        "R²": "{:.3f}",
                    }),
                    use_container_width=True,
                )

            st.markdown("""
            **分析提示**：
            - 大户型预测误差通常更大（样本少、户型差异大）
            - 老旧组屋定价因素更复杂（翻新、位置溢价）
            - MRT 沿线预测精度更高（价格主要由交通因素决定）
            """)

    # ---- 特征重要性 ----
    st.markdown("---")
    st.subheader("📊 特征重要性排名")

    importance_df, imp_type = get_feature_importance(model, used_features)
    if len(importance_df) > 0:
        top_n = min(15, len(importance_df))
        top_features = importance_df.head(top_n)

        st.caption(f"重要性类型: {imp_type}")
        fig = px.bar(
            top_features,
            x="重要性", y="特征",
            orientation="h",
            title=f"特征重要性 Top {top_n}",
            color="重要性",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # ---- 房价预估器 ----
    st.markdown("---")
    st.subheader("🏠 组屋价格预估器")

    with st.form("price_estimator"):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_area = st.number_input("面积（㎡）", min_value=30, max_value=200, value=95)
            input_type = st.selectbox("房型", options=sorted(df["flat_type"].unique()))
            input_storey = st.slider("楼层", min_value=1, max_value=50, value=10)

        with col2:
            input_lease = st.slider("剩余租约（年）", min_value=40, max_value=99, value=75)
            input_age = st.slider("房龄（年）", min_value=0, max_value=60, value=10)
            input_town = st.selectbox("镇区", options=sorted(df["town"].unique()))

        with col3:
            input_mature = st.radio("镇区类型",
                                    options=["成熟区", "非成熟区"],
                                    index=0)
            input_near_mrt = st.radio("近 MRT",
                                      options=["是 (<500m)", "否"],
                                      index=0)

        submitted = st.form_submit_button("🔮 预估价格", type="primary")

    if submitted:
        # 构建输入特征
        flat_type_map = {
            "2 ROOM": 2, "3 ROOM": 3, "4 ROOM": 4,}
