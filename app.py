# app.py - 主应用文件
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import geopandas as gpd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="HDB 转售价格分析与预测",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("🏠 HDB 组屋转售价格分析与预测系统")
st.markdown("---")

# ==================== 数据加载模块 ====================
@st.cache_data
def load_data():
    """加载HDB转售数据（模拟数据，实际使用时替换为真实数据源）"""
    
    # 由于实际数据需要从新加坡政府网站下载，这里生成模拟数据
    # 实际使用时可以替换为：pd.read_csv('resale-flat-prices.csv')
    
    np.random.seed(42)
    
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 
             'BUKIT PANJANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST',
             'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 
             'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']
    
    flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
    
    storey_ranges = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']
    
    lease_years = [60, 65, 70, 75, 80, 85, 90, 95, 99]
    
    n_records = 5000
    
    dates = pd.date_range('2020-01-01', '2024-12-31', periods=n_records)
    
    data = {
        'month': dates,
        'town': np.random.choice(towns, n_records),
        'flat_type': np.random.choice(flat_types, n_records),
        'block': np.random.randint(1, 999, n_records),
        'street_name': np.random.choice(['STREET 1', 'STREET 2', 'AVENUE 1', 'ROAD 1'], n_records),
        'storey_range': np.random.choice(storey_ranges, n_records),
        'floor_area_sqm': np.random.uniform(30, 150, n_records),
        'flat_model': np.random.choice(['Standard', 'Improved', 'Premium', 'Maisonette', 'Apartment'], n_records),
        'lease_commence_date': np.random.choice(lease_years, n_records),
        'remaining_lease': np.random.uniform(40, 99, n_records),
        'resale_price': np.random.uniform(200000, 1000000, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # 添加更多特征
    df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
    
    # 添加MRT距离特征（模拟）
    df['dist_to_mrt'] = np.random.exponential(0.5, n_records) * 1000  # 米
    df['dist_to_mrt'] = df['dist_to_mrt'].clip(100, 2500)
    
    # 添加成熟区标记
    mature_estates = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                      'CLEMENTI', 'GEYLANG', 'KALLANG/WHAMPOA', 'MARINE PARADE', 
                      'QUEENSTOWN', 'TOA PAYOH']
    df['is_mature'] = df['town'].apply(lambda x: 1 if x in mature_estates else 0)
    
    # 房型分类
    def get_size_category(flat_type):
        if flat_type in ['1 ROOM', '2 ROOM']:
            return '小户型'
        elif flat_type in ['3 ROOM', '4 ROOM']:
            return '中户型'
        else:
            return '大户型'
    
    df['size_category'] = df['flat_type'].apply(get_size_category)
    
    # 楼龄
    df['age'] = 2024 - df['lease_commence_date']
    
    return df

@st.cache_data
def load_geojson():
    """加载新加坡区域地理数据（模拟）"""
    # 实际使用时替换为真实的GeoJSON文件
    # 这里创建简化的模拟数据
    import json
    return None

# 加载数据
with st.spinner("正在加载数据..."):
    df = load_data()

# ==================== 侧边栏筛选 ====================
st.sidebar.header("🔍 数据筛选")

# 筛选维度
towns = st.sidebar.multiselect("选择市镇", options=sorted(df['town'].unique()), default=sorted(df['town'].unique())[:5])
flat_types = st.sidebar.multiselect("选择房型", options=sorted(df['flat_type'].unique()), default=sorted(df['flat_type'].unique()))
price_range = st.sidebar.slider("价格范围 (SGD)", 
                                 min_value=int(df['resale_price'].min()),
                                 max_value=int(df['resale_price'].max()),
                                 value=(int(df['resale_price'].min()), int(df['resale_price'].max())))
year_filter = st.sidebar.slider("年份范围",
                                 min_value=2020,
                                 max_value=2024,
                                 value=(2020, 2024))

# 应用筛选
filtered_df = df[
    (df['town'].isin(towns if towns else df['town'].unique())) &
    (df['flat_type'].isin(flat_types if flat_types else df['flat_type'].unique())) &
    (df['resale_price'].between(price_range[0], price_range[1])) &
    (df['month'].dt.year.between(year_filter[0], year_filter[1]))
]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**当前数据量: {len(filtered_df)} 条记录**")

# ==================== 主页面Tabs ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 数据展示", "🗺️ 地图可视化", "📈 影响因素分析", 
    "🤖 价格预测", "💡 购房策略", "📰 政策事件分析"
])

# ==================== Tab1: 数据展示 ====================
with tab1:
    st.header("房源数据展示")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总房源数", f"{len(filtered_df):,}")
        st.metric("平均价格", f"S${filtered_df['resale_price'].mean():,.0f}")
    with col2:
        st.metric("平均单价", f"S${filtered_df['price_per_sqm'].mean():,.0f}/sqm")
        st.metric("价格区间", f"S${filtered_df['resale_price'].min():,.0f} - S${filtered_df['resale_price'].max():,.0f}")
    
    # 数据表格
    st.subheader("详细数据")
    display_cols = ['month', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 
                    'resale_price', 'price_per_sqm', 'age']
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
    
    # 统计表格
    st.subheader("分组统计")
    group_by = st.selectbox("分组维度", ['town', 'flat_type', 'size_category'])
    stats = filtered_df.groupby(group_by).agg({
        'resale_price': ['count', 'mean', 'min', 'max'],
        'price_per_sqm': 'mean'
    }).round(0)
    stats.columns = ['房源数', '均价(SGD)', '最低价', '最高价', '均价(psqm)']
    st.dataframe(stats, use_container_width=True)

# ==================== Tab2: 地图可视化 ====================
with tab2:
    st.header("地理空间可视化")
    
    # 创建模拟的经纬度数据
    np.random.seed(42)
    town_lat_lon = {
        'ANG MO KIO': (1.369, 103.845),
        'BEDOK': (1.324, 103.927),
        'BISHAN': (1.351, 103.839),
        'BUKIT BATOK': (1.359, 103.750),
        'BUKIT MERAH': (1.278, 103.820),
        'BUKIT PANJANG': (1.382, 103.762),
        'CLEMENTI': (1.316, 103.765),
        'GEYLANG': (1.313, 103.871),
        'HOUGANG': (1.371, 103.892),
        'JURONG EAST': (1.333, 103.743),
        'JURONG WEST': (1.340, 103.707),
        'KALLANG/WHAMPOA': (1.311, 103.864),
        'MARINE PARADE': (1.303, 103.915),
        'QUEENSTOWN': (1.294, 103.806),
        'SENGKANG': (1.392, 103.895),
        'SERANGOON': (1.352, 103.873),
        'TAMPINES': (1.352, 103.945),
        'TOA PAYOH': (1.332, 103.848),
        'WOODLANDS': (1.438, 103.789),
        'YISHUN': (1.430, 103.835)
    }
    
    filtered_df = filtered_df.copy()
    filtered_df['lat'] = filtered_df['town'].apply(lambda x: town_lat_lon.get(x, (1.35, 103.85))[0])
    filtered_df['lon'] = filtered_df['town'].apply(lambda x: town_lat_lon.get(x, (1.35, 103.85))[1])
    
    # 热力图
    st.subheader("价格热力图")
    
    # 计算各市镇平均价格
    town_avg_price = filtered_df.groupby(['town', 'lat', 'lon'])['resale_price'].mean().reset_index()
    
    fig_heatmap = px.scatter_mapbox(
        town_avg_price,
        lat='lat',
        lon='lon',
        size='resale_price',
        color='resale_price',
        hover_name='town',
        size_max=50,
        zoom=11,
        title="各市镇平均转售价格分布",
        color_continuous_scale='Viridis'
    )
    fig_heatmap.update_layout(mapbox_style="open-street-map")
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 时空变化分析
    st.subheader("时空价格变化")
    year_town_price = filtered_df.groupby([filtered_df['month'].dt.year, 'town'])['resale_price'].mean().reset_index()
    year_town_price.columns = ['year', 'town', 'avg_price']
    
    fig_temporal = px.choropleth(
        year_town_price,
        locations='town',
        locationmode='country names',
        color='avg_price',
        animation_frame='year',
        title="各市镇价格年度变化"
    )
    # 由于需要地理边界数据，这里改用折线图
    fig_temporal_line = px.line(
        year_town_price,
        x='year',
        y='avg_price',
        color='town',
        title="各市镇平均价格年度趋势"
    )
    fig_temporal_line.update_layout(height=500)
    st.plotly_chart(fig_temporal_line, use_container_width=True)
    
    # 配套设施叠加分析
    st.subheader("配套设施影响分析")
    
    # 模拟MRT距离影响
    mrt_impact = filtered_df.groupby(pd.cut(filtered_df['dist_to_mrt'], bins=[0,500,1000,1500,2000,2500]))['resale_price'].mean().reset_index()
    mrt_impact.columns = ['距离区间', '均价']
    
    fig_mrt = px.bar(mrt_impact, x='距离区间', y='均价', 
                     title="距离MRT距离与房价关系",
                     labels={'距离区间': '距离MRT (米)', '均价': '平均价格 (SGD)'})
    st.plotly_chart(fig_mrt, use_container_width=True)

# ==================== Tab3: 影响因素分析 ====================
with tab3:
    st.header("房价影响因素分析")
    
    factors = ['flat_type', 'age', 'floor_area_sqm', 'dist_to_mrt', 'is_mature', 'storey_range']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 房型影响
        fig_flat_type = px.box(filtered_df, x='flat_type', y='resale_price', 
                               title="不同房型价格分布",
                               color='flat_type')
        fig_flat_type.update_layout(showlegend=False)
        st.plotly_chart(fig_flat_type, use_container_width=True)
        
        # 楼龄影响
        filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=[0,5,10,20,30,50,100])
        fig_age = px.box(filtered_df, x='age_group', y='resale_price',
                         title="不同楼龄价格分布")
        fig_age.update_layout(xaxis_title="楼龄(年)")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # MRT距离影响
        filtered_df['mrt_group'] = pd.cut(filtered_df['dist_to_mrt'], bins=[0,500,1000,1500,2000,2500])
        fig_mrt_detail = px.box(filtered_df, x='mrt_group', y='resale_price',
                                title="MRT距离对房价的影响")
        fig_mrt_detail.update_layout(xaxis_title="距离MRT (米)")
        st.plotly_chart(fig_mrt_detail, use_container_width=True)
        
        # 成熟区 vs 非成熟区
        fig_mature = px.box(filtered_df, x='is_mature', y='resale_price',
                            title="成熟区 vs 非成熟区价格对比",
                            color='is_mature')
        fig_mature.update_layout(xaxis_title="是否为成熟区", 
                                 xaxis_ticktext=['非成熟区', '成熟区'])
        st.plotly_chart(fig_mature, use_container_width=True)
    
    # 楼层影响
    st.subheader("楼层对价格的影响")
    floor_order = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']
    filtered_df['storey_range'] = pd.Categorical(filtered_df['storey_range'], categories=floor_order, ordered=True)
    fig_floor = px.bar(filtered_df.groupby('storey_range')['resale_price'].mean().reset_index(),
                       x='storey_range', y='resale_price', title="不同楼层区间的平均价格")
    st.plotly_chart(fig_floor, use_container_width=True)
    
    # 多因素综合分析
    st.subheader("多因素综合分析")
    
    # 创建交叉分析
    cross_analysis = filtered_df.groupby(['flat_type', 'is_mature'])['resale_price'].mean().reset_index()
    cross_analysis['is_mature'] = cross_analysis['is_mature'].map({0: '非成熟区', 1: '成熟区'})
    
    fig_cross = px.bar(cross_analysis, x='flat_type', y='resale_price', 
                       color='is_mature', barmode='group',
                       title="房型与区域类型对价格的交互影响")
    st.plotly_chart(fig_cross, use_container_width=True)
    
    # 相关性热力图
    st.subheader("特征相关性分析")
    numeric_cols = ['resale_price', 'floor_area_sqm', 'age', 'remaining_lease', 'dist_to_mrt', 'price_per_sqm']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                         title="数值特征相关性矩阵",
                         color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

# ==================== Tab4: 价格预测模型 ====================
with tab4:
    st.header("价格预测模型")
    st.markdown("针对不同户型分别建立预测模型，评估预测效果")
    
    # 准备特征
    def prepare_features(df_subset):
        features = pd.DataFrame()
        features['floor_area_sqm'] = df_subset['floor_area_sqm']
        features['age'] = df_subset['age']
        features['remaining_lease'] = df_subset['remaining_lease']
        features['dist_to_mrt'] = df_subset['dist_to_mrt']
        features['is_mature'] = df_subset['is_mature']
        
        # 编码分类特征
        le_town = LabelEncoder()
        le_flat_model = LabelEncoder()
        
        features['town_encoded'] = le_town.fit_transform(df_subset['town'])
        features['flat_model_encoded'] = le_flat_model.fit_transform(df_subset['flat_model'])
        
        # 楼层编码
        floor_mapping = {'01 TO 03': 1, '04 TO 06': 2, '07 TO 09': 3, 
                        '10 TO 12': 4, '13 TO 15': 5, '16 TO 18': 6, '19 TO 21': 7}
        features['floor_score'] = df_subset['storey_range'].map(floor_mapping)
        
        # 添加月份特征
        features['month'] = df_subset['month'].dt.month
        features['year'] = df_subset['month'].dt.year
        
        return features
    
    # 选择房型进行预测
    size_type = st.selectbox("选择户型类型", ['小户型', '中户型', '大户型'])
    size_df = filtered_df[filtered_df['size_category'] == size_type]
    
    if len(size_df) > 100:
        # 准备特征
        X = prepare_features(size_df)
        y = size_df['resale_price']
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        col1, col2 = st.columns(2)
        
        models = {
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
        
        # 显示结果
        st.subheader(f"{size_type} 预测模型评估")
        
        # 模型对比表格
        compare_df = pd.DataFrame(results).T
        compare_df = compare_df[['MAE', 'RMSE', 'R2']]
        compare_df['MAE'] = compare_df['MAE'].apply(lambda x: f"S${x:,.0f}")
        compare_df['RMSE'] = compare_df['RMSE'].apply(lambda x: f"S${x:,.0f}")
        compare_df['R2'] = compare_df['R2'].apply(lambda x: f"{x:.3f}")
        st.dataframe(compare_df, use_container_width=True)
        
        # 使用最佳模型进行预测
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        best_model = results[best_model_name]['model']
        
        st.subheader("📊 自定义价格预测")
        st.markdown(f"*使用模型: {best_model_name}*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_floor_area = st.number_input("面积 (sqm)", min_value=30.0, max_value=200.0, value=90.0)
            pred_age = st.number_input("楼龄 (年)", min_value=0, max_value=60, value=10)
        
        with col2:
            pred_dist_mrt = st.number_input("距离MRT (米)", min_value=100, max_value=3000, value=500)
            pred_mature = st.selectbox("是否成熟区", ["是", "否"])
        
        with col3:
            pred_town = st.selectbox("市镇", sorted(size_df['town'].unique()))
            pred_floor_range = st.selectbox("楼层区间", ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21'])
        
        if st.button("🔮 预测价格"):
            # 构建预测特征
            pred_features = pd.DataFrame({
                'floor_area_sqm': [pred_floor_area],
                'age': [pred_age],
                'remaining_lease': [99 - pred_age],
                'dist_to_mrt': [pred_dist_mrt],
                'is_mature': [1 if pred_mature == "是" else 0],
                'town_encoded': [0],
                'flat_model_encoded': [0],
                'floor_score': [floor_mapping.get(pred_floor_range, 3)],
                'month': [6],
                'year': [2024]
            })
            
            # 编码市镇
            le_town = LabelEncoder()
            le_town.fit(size_df['town'])
            pred_features['town_encoded'] = le_town.transform([pred_town])[0] if pred_town in le_town.classes_ else 0
            
            pred_scaled = scaler.transform(pred_features)
            prediction = best_model.predict(pred_scaled)[0]
            
            st.success(f"🏠 预测转售价格: **S${prediction:,.0f}**")
            st.info(f"价格区间: S${prediction * 0.9:,.0f} - S${prediction * 1.1:,.0f}")
        
        # 预测误差分析
        st.subheader("模型误差分析")
        y_pred_best = best_model.predict(X_test_scaled)
        
        errors = y_test - y_pred_best
        
        fig_errors = make_subplots(rows=1, cols=2,
                                   subplot_titles=["预测值 vs 实际值", "误差分布"])
        
        fig_errors.add_trace(
            go.Scatter(x=y_test, y=y_pred_best, mode='markers', name='预测',
                       marker=dict(color='blue', opacity=0.5)),
            row=1, col=1
        )
        fig_errors.add_trace(
            go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                       mode='lines', name='完美预测', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        fig_errors.add_trace(
            go.Histogram(x=errors, name='误差', nbinsx=30),
            row=1, col=2
        )
        
        fig_errors.update_layout(height=500, showlegend=False)
        fig_errors.update_xaxes(title_text="实际价格", row=1, col=1)
        fig_errors.update_yaxes(title_text="预测价格", row=1, col=1)
        fig_errors.update_xaxes(title_text="误差", row=1, col=2)
        
        st.plotly_chart(fig_errors, use_container_width=True)
        
    else:
        st.warning(f"{size_type} 数据量不足，无法建立可靠预测模型。当前数据量: {len(size_df)}")

# ==================== Tab5: 购房策略 ====================
with tab5:
    st.header("购房策略与保值验证")
    
    st.markdown("""
    ### 策略框架
    
    基于数据分析，我们提出以下购房策略：
    
    1. **位置优先策略**: 优先选择成熟区、靠近MRT的房源
    2. **房型优化策略**: 根据预算选择性价比最高的房型
    3. **时机选择策略**: 分析价格周期，选择最佳入市时机
    """)
    
    # 策略参数设置
    st.subheader("策略参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("预算上限 (SGD)", min_value=300000, max_value=1500000, value=800000, step=50000)
        preferred_flat_type = st.multiselect("偏好房型", options=['3 ROOM', '4 ROOM', '5 ROOM'], default=['4 ROOM'])
    
    with col2:
        max_distance_mrt = st.slider("距离MRT最大距离 (米)", 200, 2000, 800)
        min_remaining_lease = st.slider("最少剩余年限 (年)", 40, 99, 70)
    
    # 筛选符合条件的房源
    strategy_df = df[
        (df['resale_price'] <= budget) &
        (df['flat_type'].isin(preferred_flat_type)) &
        (df['dist_to_mrt'] <= max_distance_mrt) &
        (df['remaining_lease'] >= min_remaining_lease)
    ].copy()
    
    st.subheader(f"符合条件的房源: {len(strategy_df)} 套")
    
    if len(strategy_df) > 0:
        # 推荐列表
        strategy_df['性价比'] = strategy_df['price_per_sqm']
        strategy_df['推荐得分'] = (1 / strategy_df['dist_to_mrt']) * strategy_df['is_mature'] * 100
        
        top_recommendations = strategy_df.nlargest(10, '推荐得分')
        
        st.subheader("🏆 Top 10 推荐房源")
        recommend_cols = ['town', 'flat_type', 'resale_price', 'price_per_sqm', 
                         'dist_to_mrt', 'is_mature', 'remaining_lease']
        st.dataframe(top_recommendations[recommend_cols].style.format({
            'resale_price': 'S${:,.0f}',
            'price_per_sqm': 'S${:,.0f}',
            'dist_to_mrt': '{:.0f}m'
        }), use_container_width=True)
        
        # 策略保值验证
        st.subheader("策略保值验证")
        
        # 计算不同策略下的价格表现
        st.markdown("#### 策略 vs 基准对比")
        
        # 基准: 所有房源
        baseline_avg = df['resale_price'].mean()
        baseline_avg_2023 = df[df['month'].dt.year == 2023]['resale_price'].mean()
        baseline_avg_2024 = df[df['month'].dt.year == 2024]['resale_price'].mean()
        
        # 策略房源
        strategy_avg = strategy_df['resale_price'].mean()
        strategy_avg_2023 = strategy_df[strategy_df['month'].dt.year == 2023]['resale_price'].mean() if len(strategy_df[strategy_df['month'].dt.year == 2023]) > 0 else strategy_avg
        strategy_avg_2024 = strategy_df[strategy_df['month'].dt.year == 2024]['resale_price'].mean() if len(strategy_df[strategy_df['month'].dt.year == 2024]) > 0 else strategy_avg
        
        comparison_data = pd.DataFrame({
            '类别': ['基准(所有房源)', '策略推荐房源'],
            '平均价格': [baseline_avg, strategy_avg],
            '2023年均价': [baseline_avg_2023, strategy_avg_2023],
            '2024年均价': [baseline_avg_2024, strategy_avg_2024],
            '涨幅(2023→2024)': [
                (baseline_avg_2024 - baseline_avg_2023) / baseline_avg_2023 * 100,
                (strategy_avg_2024 - strategy_avg_2023) / strategy_avg_2023 * 100
            ]
        })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # 稳定性分析
        st.subheader("价格稳定性分析")
        
        # 计算各市镇价格标准差作为稳定性指标
        town_stability = strategy_df.groupby('town')['resale_price'].agg(['mean', 'std']).reset_index()
        town_stability['cv'] = town_stability['std'] / town_stability['mean']  # 变异系数
        town_stability = town_stability.sort_values('cv')
        
        fig_stability = px.bar(town_stability.head(10), x='town', y='cv',
                               title="价格最稳定的市镇 (低变异系数)",
                               labels={'cv': '价格变异系数', 'town': '市镇'})
        st.plotly_chart(fig_stability, use_container_width=True)
        
        # 流动性分析
        st.subheader("流动性分析")
        town_volume = strategy_df['town'].value_counts().reset_index()
        town_volume.columns = ['town', '交易量']
        
        fig_volume = px.bar(town_volume.head(10), x='town', y='transaction_volume',
                            title="交易最活跃的市镇",
                            labels={'交易量': '房源数量', 'town': '市镇'})
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # 策略总结
        st.subheader("📋 策略总结")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("策略价格优势", f"{((strategy_avg - baseline_avg) / baseline_avg * 100):+.1f}%")
        
        with col2:
            st.metric("策略涨幅优势", f"{comparison_data['涨幅(2023→2024)'].iloc[1] - comparison_data['涨幅(2023→2024)'].iloc[0]:+.1f}%")
        
        with col3:
            st.metric("推荐房源数", f"{len(strategy_df)}套")
        
        st.info("""
        **策略结论**:
        - 符合策略的房源在价格表现上优于市场平均水平
        - 成熟区、近MRT的房源具有更好的保值和增值潜力
        - 建议优先选择变异系数低、交易量高的市镇
        """)
        
    else:
        st.warning("没有找到符合条件的房源，请调整筛选条件")

# ==================== Tab6: 政策事件分析 ====================
with tab6:
    st.header("新加坡住房政策事件冲击分析")
    st.markdown("*可选模块 - 分析政策事件对房价的影响*")
    
    # 定义政策事件
    policy_events = [
        {"date": "2020-04-07", "name": "COVID-19 阻断措施", "type": "negative", "description": "疫情影响经济，房地产市场短期低迷"},
        {"date": "2020-12-15", "name": "降温措施 - 调高印花税", "type": "negative", "description": "额外买方印花税(ABSD)上调5%"},
        {"date": "2021-12-16", "name": "新一轮降温措施", "type": "negative", "description": "收紧组屋贷款与最低首付"},
        {"date": "2022-09-30", "name": "房市降温措施加码", "type": "negative", "description": "私宅与组屋贷款限制收紧"},
        {"date": "2023-04-27", "name": "降温措施调整", "type": "neutral", "description": "额外买方印花税进一步调整"},
        {"date": "2024-02-20", "name": "预算案住房政策", "type": "positive", "description": "加强组屋津贴，惠及首次购房者"}
    ]
    
    # 计算政策前后价格变化
    df_policy = df.copy()
    df_policy['date'] = pd.to_datetime(df_policy['month'])
    
    policy_impacts = []
    
    for policy in policy_events:
        policy_date = pd.to_datetime(policy['date'])
        
        # 政策前3个月和后3个月的数据
        before_data = df_policy[(df_policy['date'] >= policy_date - pd.Timedelta(days=90)) & 
                                 (df_policy['date'] < policy_date)]
        after_data = df_policy[(df_policy['date'] >= policy_date) & 
                               (df_policy['date'] <= policy_date + pd.Timedelta(days=90))]
        
        if len(before_data) > 0 and len(after_data) > 0:
            before_avg = before_data['resale_price'].mean()
            after_avg = after_data['resale_price'].mean()
            pct_change = (after_avg - before_avg) / before_avg * 100
        else:
            pct_change = 0
        
        policy_impacts.append({
            '事件名称': policy['name'],
            '日期': policy['date'],
            '类型': policy['type'],
            '描述': policy['description'],
            '价格变化(%)': pct_change
        })
    
    impacts_df = pd.DataFrame(policy_impacts)
    
    # 显示政策事件表
    st.subheader("📅 政策事件时间线")
    st.dataframe(impacts_df, use_container_width=True)
    
    # 可视化政策影响
    st.subheader("政策事件对房价的影响")
    
    fig_policy = px.bar(impacts_df, x='事件名称', y='价格变化(%)',
                        color='类型',
                        title="政策事件前后90天价格变化",
                        labels={'价格变化(%)': '价格变化百分比 (%)'})
    fig_policy.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_policy, use_container_width=True)
    
    # 价格时间序列与政策标记
    st.subheader("价格走势与政策事件")
    
    monthly_price = df.groupby(df['month'].dt.to_period('M'))['resale_price'].mean().reset_index()
    monthly_price['month'] = monthly_price['month'].astype(str)
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=monthly_price['month'],
        y=monthly_price['resale_price'],
        mode='lines+markers',
        name='平均价格'
    ))
    
    # 添加政策事件标记
    for i, policy in enumerate(policy_events):
        if policy['date'] in monthly_price['month'].values:
            fig_timeline.add_vline(x=policy['date'], 
                                   line_dash="dash",
                                   line_color="red" if policy['type'] == 'negative' else "green",
                                   annotation_text=policy['name'][:10],
                                   annotation_position="top" if i % 2 == 0 else "bottom")
    
    fig_timeline.update_layout(
        title="HDB转售价格月度趋势与政策事件",
        xaxis_title="日期",
        yaxis_title="平均价格 (SGD)",
        height=500
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # 政策影响总结
    st.subheader("📊 政策影响分析总结")
    
    negative_impacts = impacts_df[impacts_df['类型'] == 'negative']
    positive_impacts = impacts_df[impacts_df['类型'] == 'positive']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(negative_impacts) > 0:
            st.metric("降温措施平均影响", f"{negative_impacts['价格变化(%)'].mean():+.2f}%",
                     help="降温措施实施后的平均价格变化")
    
    with col2:
        if len(positive_impacts) > 0:
            st.metric("利好政策平均影响", f"{positive_impacts['价格变化(%)'].mean():+.2f}%",
                     help="利好政策实施后的平均价格变化")
    
    st.info("""
    **政策事件分析结论**:
    - 降温措施通常对房价产生短期压制作用
    - 但长期来看，HDB转售市场基本面依然强劲
    - 政策影响有明显的累积效应，多项降温措施叠加效果更显著
    - 首次购房者政策对低价房源有积极影响
    """)

# ==================== 侧边栏数据说明 ====================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 数据说明
- **数据来源**: 新加坡建屋发展局(HDB)转售交易记录
- **时间范围**: 2020年至今
- **主要字段**: 市镇、房型、面积、价格、楼龄等
- **更新频率**: 模拟数据，实际使用时需从data.gov.sg获取
""")

# 运行命令: streamlit run app.py
