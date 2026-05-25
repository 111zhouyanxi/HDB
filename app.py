import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pydeck as pdk

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

    # 时间处理
    df['month'] = pd.to_datetime(df['month'])

    df['year'] = df['month'].dt.year

    # 房龄
    df['house_age'] = 2026 - df['lease_commence_date']

    # 提取剩余租约
    def extract_lease(x):
        match = re.search(r'(\d+)', str(x))
        if match:
            return int(match.group(1))
        return np.nan

    df['remaining_lease_year'] = df['remaining_lease'].apply(extract_lease)

    # 单位面积价格
    df['unit_price'] = df['resale_price'] / df['floor_area_sqm']

    return df

df = load_data()

# =====================================================
# 侧边栏
# =====================================================

st.sidebar.title("功能菜单")

menu = st.sidebar.radio(
    "请选择功能",
    [
        "数据概览",
        "房价趋势分析",
        "镇区分析",
        "地图可视化",
        "影响因素分析",
        "房价预测"
    ]
)
# =====================================================
# 数据概览
# =====================================================

if menu == "数据概览":

    st.header("HDB 数据筛选与统计分析")

    # =================================================
    # 侧边栏筛选
    # =================================================

    st.sidebar.header("数据筛选")

    # 镇区选择
    towns = sorted(df["town"].unique())

    selected_towns = st.sidebar.multiselect(
        "选择镇区",
        options=towns,
        placeholder="请选择镇区",
        label_visibility="visible"
    )

    # 年份范围
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    year_range = st.sidebar.slider(
        "年份范围",
        min_year,
        max_year,
        (2020, max_year)
    )

    # 房型选择
    flat_type = st.sidebar.selectbox(
        "房型",
        [
            "不限",
            "3 ROOM",
            "4 ROOM",
            "5 ROOM",
            "EXECUTIVE"
        ]
    )

    # 面积范围
    min_area = int(df["floor_area_sqm"].min())
    max_area = int(df["floor_area_sqm"].max())

    area_range = st.sidebar.slider(
        "面积范围（㎡）",
        min_area,
        max_area,
        (60, 120)
    )

    # =================================================
    # 数据筛选
    # =================================================
    # 如果没有选择镇区，默认全部
    if len(selected_towns) == 0:
        selected_towns = towns
    filtered_df = df[
        (df["town"].isin(selected_towns)) &
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1]) &
        (df["floor_area_sqm"] >= area_range[0]) &
        (df["floor_area_sqm"] <= area_range[1])
    ]
    # 房型筛选
    if flat_type != "不限":

        filtered_df = filtered_df[
            filtered_df["flat_type"] == flat_type
        ]

    # =================================================
    # 统计指标
    # =================================================

    total_count = len(filtered_df)

    avg_price = filtered_df["unit_price"].mean()

    avg_total = filtered_df["resale_price"].mean()

    max_price = filtered_df["unit_price"].max()

    min_price = filtered_df["unit_price"].min()

    # =================================================
    # 指标看板
    # =================================================

    st.subheader("基础统计看板")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "成交套数",
        f"{total_count} 套"
    )

    col2.metric(
        "平均单价",
        f"S${avg_price:.0f}/㎡"
    )

    col3.metric(
        "平均总价",
        f"S${avg_total:.0f}"
    )

    col4.metric(
        "最高单价",
        f"S${max_price:.0f}/㎡"
    )

    # =================================================
    # 最低单价
    # =================================================

    st.metric(
        "最低单价",
        f"S${min_price:.0f}/㎡"
    )

    # =================================================
    # 数据表格
    # =================================================

    st.subheader("筛选后的成交记录")


    st.dataframe(

        filtered_df[
            [
                "month",
                "town",
                "flat_type",
                "floor_area_sqm",
                "resale_price",
                "unit_price"
            ]
        ]
    )

    # =================================================
    # 图表展示
    # =================================================

    st.subheader("镇区平均房价")

    # 判断是否有数据
    if filtered_df.empty:

        st.warning("当前筛选条件下没有数据，请重新选择条件")

    else:

        town_avg = filtered_df.groupby("town")[
            "resale_price"
        ].mean()

        fig, ax = plt.subplots(figsize=(10, 6))

        town_avg.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("各镇区平均房价")

        ax.set_ylabel("平均房价")

        st.pyplot(fig)
# =====================================================
# 房价趋势分析
# =====================================================

elif menu == "房价趋势分析":

    st.header("房价趋势分析")

    yearly_price = df.groupby('year')['resale_price'].mean()

    fig, ax = plt.subplots(figsize=(10,6))

    yearly_price.plot(marker='o', ax=ax)

    ax.set_title("年度平均房价趋势")

    ax.set_xlabel("年份")

    ax.set_ylabel("平均房价")

    ax.grid()

    st.pyplot(fig)

# =====================================================
# 镇区分析
# =====================================================

elif menu == "镇区分析":

    st.header("镇区均价分析")

    town_price = df.groupby('town')['resale_price'].mean()

    town_price = town_price.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12,6))

    town_price.head(10).plot(kind='bar', ax=ax)

    ax.set_title("均价最高的10个镇区")

    ax.set_ylabel("平均房价")

    st.pyplot(fig)

# =====================================================
# 影响因素分析
# =====================================================

elif menu == "影响因素分析":

    st.header("房价影响因素分析")

    factor = st.selectbox(
        "选择分析因素",
        [
            "面积与价格",
            "房龄与价格",
            "剩余租约与价格"
        ]
    )

    fig, ax = plt.subplots(figsize=(8,6))

    if factor == "面积与价格":

        sns.scatterplot(
            x='floor_area_sqm',
            y='unit_price',
            data=df,
            ax=ax
        )

        ax.set_title("面积与单价关系")

    elif factor == "房龄与价格":

        sns.scatterplot(
            x='house_age',
            y='unit_price',
            data=df,
            ax=ax
        )

        ax.set_title("房龄与单价关系")

    elif factor == "剩余租约与价格":

        sns.scatterplot(
            x='remaining_lease_year',
            y='unit_price',
            data=df,
            ax=ax
        )

        ax.set_title("剩余租约与单价关系")

    st.pyplot(fig)

# =====================================================
# 地图可视化
# =====================================================

elif menu == "地图可视化":

    st.header("地图可视化分析")

    map_type = st.selectbox(
        "选择地图类型",
        [
            "镇区均价地图",
            "房价热力地图",
            "房价动态地图"
        ]
    )

    # =================================================
    # 模拟镇区经纬度
    # =================================================

    town_coords = {
        "ANG MO KIO": [1.3691, 103.8454],
        "BEDOK": [1.3236, 103.9273],
        "BISHAN": [1.3500, 103.8485],
        "BUKIT BATOK": [1.3496, 103.7528],
        "BUKIT MERAH": [1.2770, 103.8190],
        "CHOA CHU KANG": [1.3854, 103.7443],
        "CLEMENTI": [1.3151, 103.7650],
        "GEYLANG": [1.3188, 103.8870],
        "HOUGANG": [1.3612, 103.8863],
        "JURONG EAST": [1.3329, 103.7436],
        "JURONG WEST": [1.3404, 103.7060],
        "KALLANG/WHAMPOA": [1.3106, 103.8660],
        "PASIR RIS": [1.3735, 103.9493],
        "PUNGGOL": [1.4043, 103.9020],
        "QUEENSTOWN": [1.2942, 103.7861],
        "SENGKANG": [1.3919, 103.8957],
        "TAMPINES": [1.3496, 103.9568],
        "TOA PAYOH": [1.3343, 103.8563],
        "WOODLANDS": [1.4382, 103.7890],
        "YISHUN": [1.4294, 103.8354]
    }

    # 添加经纬度
    df['latitude'] = df['town'].map(
        lambda x: town_coords.get(x, [1.35, 103.82])[0]
    )

    df['longitude'] = df['town'].map(
        lambda x: town_coords.get(x, [1.35, 103.82])[1]
    )

    # =================================================
    # 1. 镇区均价地图
    # =================================================

    if map_type == "镇区均价地图":

        town_avg = df.groupby(
            ['town', 'latitude', 'longitude']
        )['unit_price'].mean().reset_index()

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=town_avg,
            get_position='[longitude, latitude]',
            get_radius='unit_price / 10',
            get_fill_color='[255, 140, 0, 160]',
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=10
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": "{town}\n平均单价: {unit_price}"
            }
        )

        st.pydeck_chart(deck)

    # =================================================
    # 2. 热力地图
    # =================================================

    elif map_type == "房价热力地图":

        radius = st.slider(
            "热力图半径",
            20,
            100,
            50
        )
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_weight='unit_price',
            radius_pixels=radius
        )
        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=10
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state
        )

        st.pydeck_chart(deck)

    # =================================================
    # 3. 动态地图
    # =================================================

    elif map_type == "房价动态地图":

        years = sorted(df['year'].unique())

        selected_year = st.select_slider(
            "选择年份",
            options=years
        )

        year_df = df[df['year'] == selected_year]

        town_avg = year_df.groupby(
            ['town', 'latitude', 'longitude']
        )['unit_price'].mean().reset_index()

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=town_avg,
            get_position='[longitude, latitude]',
            get_radius='unit_price / 10',
            get_fill_color='[0, 128, 255, 180]',
            pickable=True
        )

        view_state = pdk.ViewState(
            latitude=1.3521,
            longitude=103.8198,
            zoom=10
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": "{town}\n年份: " + str(selected_year)
            }
        )

        st.pydeck_chart(deck)


# =====================================================
# 房价预测
# =====================================================

elif menu == "房价预测":

    st.header("HDB 房价预测")

    # 特征
    features = [
        'floor_area_sqm',
        'house_age',
        'remaining_lease_year'
    ]

    X = df[features]

    y = df['resale_price']

    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    score = r2_score(y_test, y_pred)

    st.success(f"模型 R² 得分：{score:.4f}")

    st.subheader("请输入房屋信息")

    area = st.slider(
        "房屋面积（平方米）",
        30,
        200,
        90
    )

    age = st.slider(
        "房龄",
        1,
        60,
        20
    )

    lease = st.slider(
        "剩余租约",
        1,
        99,
        75
    )

    if st.button("开始预测"):

        sample = [[area, age, lease]]

        result = model.predict(sample)

        st.balloons()

        st.success(
            f"预测房价：S${result[0]:,.0f}"
        )
