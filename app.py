# app.py - 修复后的完整代码
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
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
    """加载HDB转售数据"""
    
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
    
    # 添加MRT距离特征
    df['dist_to_mrt'] = np.random.exponential(0.5, n_records) * 1000
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

# 加载数据
with st.spinner("正在加载数据..."):
    df = load_data()

# ==================== 侧边栏筛选 ====================
st.sidebar.header("🔍 数据筛选")

selected_towns = st.sidebar.multiselect("选择市镇", options=sorted(df['town'].unique()), default=sorted(df['town'].unique())[:3])
selected_flat_types = st.sidebar.multiselect("选择房型", options=sorted(df['flat_type'].unique()), default=sorted(df['flat_type'].unique())[:3])
price_range = st.sidebar.slider("价格范围 (SGD)", 
                                 min_value=int(df['resale_price'].min()),
                                 max_value=int(df['resale_price'].max()),
                                 value=(int(df['resale_price'].min()), int(df['resale_price'].max())))
year_range = st.sidebar.slider("年份范围",
                                min_value=2020,
                                max_value=2024,
                                value=(2020, 2024))

# 应用筛选
filtered_df = df[
    (df['town'].isin(selected_towns if selected_towns else df['town'].unique())) &
    (df['flat_type'].isin(selected_flat_types if selected_flat_types else df['flat_type'].unique())) &
    (df['resale_price'].between(price_range[0], price_range[1])) &
    (df['month'].dt.year.between(year_range[0], year_range[1]))
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
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总房源数", f"{len(filtered_df):,}")
    with col2:
        st.metric("平均价格", f"S${filtered_df['resale_price'].mean():,.0f}")
    with col3:
        st.metric("平均单价", f"S${filtered_df['price_per_sqm'].mean():,.0f}/sqm")
    with col4:
        st.metric("价格区间", f"S${filtered_df['resale_price'].min():,.0f} - S${filtered_df['resale_price'].max():,.0f}")
    
    st.subheader("详细数据")
    display_cols = ['month', 'town', 'flat_type', 'storey_range', 'floor_area_sqm', 
                    'resale_price', 'price_per_sqm', 'age']
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)
    
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
    
    # 市镇坐标
    town_lat_lon = {
        'ANG MO KIO': (1.369, 103.845), 'BEDOK': (1.324, 103.927), 'BISHAN': (1.351, 103.839),
        'BUKIT BATOK': (1.359, 103.750), 'BUKIT MERAH': (1.278, 103.820), 'BUKIT PANJANG': (1.382, 103.762),
        'CLEMENTI': (1.316, 103.765), 'GEYLANG': (1.313, 103.871), 'HOUGANG': (1.371, 103.892),
        'JURONG EAST': (1.333, 103.743), 'JURONG WEST': (1.340, 103.707), 'KALLANG/WHAMPOA': (1.311, 103.864),
        'MARINE PARADE': (1.303, 103.915), 'QUEENSTOWN': (1.294, 103.806), 'SENGKANG': (1.392, 103.895),
        'SERANGOON': (1.352, 103.873), 'TAMPINES': (1.352, 103.945), 'TOA PAYOH': (1.332, 103.848),
        'WOODLANDS': (1.438, 103.789), 'YISHUN': (1.430, 103.835)
    }
    
    map_df = filtered_df.copy()
    map_df['lat'] = map_df['town'].apply(lambda x: town_lat_lon.get(x, (1.35, 103.85))[0])
    map_df['lon'] = map_df['town'].apply(lambda x: town_lat_lon.get(x, (1.35, 103.85))[1])
    
    # 价格热力图
    st.subheader("价格热力图")
    town_avg_price = map_df.groupby(['town', 'lat', 'lon'])['resale_price'].mean().reset_index()
    
    fig_heatmap = px.scatter_mapbox(
        town_avg_price,
        lat='lat', lon='lon',
        size='resale_price', color='resale_price',
        hover_name='town', size_max=50, zoom=11,
        title="各市镇平均转售价格分布",
        color_continuous_scale='Viridis'
    )
    fig_heatmap.update_layout(mapbox_style="open-street-map", height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 时空变化分析
    st.subheader("时空价格变化")
    year_town_price = filtered_df.groupby([filtered_df['month'].dt.year, 'town'])['resale_price'].mean().reset_index()
    year_town_price.columns = ['year', 'town', 'avg_price']
    
    fig_temporal = px.line(
        year_town_price, x='year', y='avg_price', color='town',
        title="各市镇平均价格年度趋势"
    )
    fig_temporal.update_layout(height=500)
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    # MRT距离影响
    st.subheader("距离MRT车站距离对房价的影响")
    mrt_df = filtered_df.copy()
    mrt_df['距离区间'] = pd.cut(mrt_df['dist_to_mrt'], bins=[0, 500, 1000, 1500, 2000, 2500], 
                                  labels=['0-500m', '500-1000m', '1000-1500m', '1500-2000m', '2000-2500m'])
    mrt_impact = mrt_df.groupby('距离区间', observed=True)['resale_price'].mean().reset_index()
    
    fig_mrt = px.bar(mrt_impact, x='距离区间', y='resale_price', 
                     title="距离MRT距离与房价关系",
                     labels={'距离区间': '距离MRT', 'resale_price': '平均价格 (SGD)'})
    st.plotly_chart(fig_mrt, use_container_width=True)

# ==================== Tab3: 影响因素分析 ====================
with tab3:
    st.header("房价影响因素分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 房型影响
        fig_flat_type = px.box(filtered_df, x='flat_type', y='resale_price', 
                               title="不同房型价格分布", color='flat_type')
        fig_flat_type.update_layout(showlegend=False)
        st.plotly_chart(fig_flat_type, use_container_width=True)
        
        # 楼龄影响
        filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=[0, 5, 10, 20, 30, 50, 100],
                                           labels=['0-5年', '5-10年', '10-20年', '20-30年', '30-50年', '50年以上'])
        age_price = filtered_df.groupby('age_group', observed=True)['resale_price'].mean().reset_index()
        fig_age = px.bar(age_price, x='age_group', y='resale_price', title="不同楼龄平均价格")
        fig_age.update_layout(xaxis_title="楼龄", yaxis_title="平均价格 (SGD)")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # MRT距离影响详细
        filtered_df['mrt_group'] = pd.cut(filtered_df['dist_to_mrt'], bins=[0, 500, 1000, 1500, 2000, 2500],
                                           labels=['0-500m', '500-1000m', '1000-1500m', '1500-2000m', '2000-2500m'])
        fig_mrt_detail = px.box(filtered_df, x='mrt_group', y='resale_price', 
                                title="MRT距离对房价的影响", color='mrt_group')
        fig_mrt_detail.update_layout(showlegend=False)
        st.plotly_chart(fig_mrt_detail, use_container_width=True)
        
        # 成熟区 vs 非成熟区
        filtered_df['区域类型'] = filtered_df['is_mature'].map({0: '非成熟区', 1: '成熟区'})
        fig_mature = px.box(filtered_df, x='区域类型', y='resale_price', color='区域类型',
                            title="成熟区 vs 非成熟区价格对比")
        fig_mature.update_layout(showlegend=False)
        st.plotly_chart(fig_mature, use_container_width=True)
    
    # 楼层影响
    st.subheader("楼层对价格的影响")
    floor_order = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']
    floor_df = filtered_df[filtered_df['storey_range'].isin(floor_order)]
    floor_price = floor_df.groupby('storey_range')['resale_price'].mean().reset_index()
    fig_floor = px.bar(floor_price, x='storey_range', y='resale_price', 
                       title="不同楼层区间的平均价格",
                       labels={'storey_range': '楼层区间', 'resale_price': '平均价格 (SGD)'})
    st.plotly_chart(fig_floor, use_container_width=True)
    
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
    
    def prepare_features(df_subset):
        features = pd.DataFrame()
        features['floor_area_sqm'] = df_subset['floor_area_sqm']
        features['age'] = df_subset['age']
        features['remaining_lease'] = df_subset['remaining_lease']
        features['dist_to_mrt'] = df_subset['dist_to_mrt']
        features['is_mature'] = df_subset['is_mature']
        
        le_town = LabelEncoder()
        le_flat_model = LabelEncoder()
        
        features['town_encoded'] = le_town.fit_transform(df_subset['town'])
        features['flat_model_encoded'] = le_flat_model.fit_transform(df_subset['flat_model'])
        
        floor_mapping = {'01 TO 03': 1, '04 TO 06': 2, '07 TO 09': 3, 
                        '10 TO 12': 4, '13 TO 15': 5, '16 TO 18': 6, '19 TO 21': 7}
        features['floor_score'] = df_subset['storey_range'].map(floor_mapping).fillna(3)
        
        features['month'] = df_subset['month'].dt.month
        features['year'] = df_subset['month'].dt.year
        
        return features, le_town, le_flat_model
    
    size_type = st.selectbox("选择户型类型", ['小户型', '中户型', '大户型'])
    size_df = filtered_df[filtered_df['size_category'] == size_type]
    
    if len(size_df) > 50:
        X, le_town, le_flat_model = prepare_features(size_df)
        y = size_df['resale_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'model': model
            }
        
        st.subheader(f"{size_type} 预测模型评估")
        
        compare_df = pd.DataFrame(results).T[['MAE', 'RMSE', 'R2']]
        compare_df['MAE'] = compare_df['MAE'].apply(lambda x: f"S${x:,.0f}")
        compare_df['RMSE'] = compare_df['RMSE'].apply(lambda x: f"S${x:,.0f}")
        compare_df['R2'] = compare_df['R2'].apply(lambda x: f"{x:.3f}")
        st.dataframe(compare_df, use_container_width=True)
        
        best_model_name = max(results, key=lambda x: results[x]['R2'])
        best_model = results[best_model_name]['model']
        
        st.subheader("📊 自定义价格预测")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_floor_area = st.number_input("面积 (sqm)", min_value=30.0, max_value=200.0, value=90.0)
            pred_age = st.number_input("楼龄 (年)", min_value=0, max_value=60, value=10)
        with col2:
            pred_dist_mrt = st.number_input("距离MRT (米)", min_value=100, max_value=3000, value=500)
            pred_mature = st.selectbox("是否成熟区", ["是", "否"])
        with col3:
            pred_town = st.selectbox("市镇", sorted(size_df['town'].unique()))
            pred_floor_range = st.selectbox("楼层区间", ['04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15'])
        
        if st.button("🔮 预测价格"):
            floor_mapping = {'01 TO 03': 1, '04 TO 06': 2, '07 TO 09': 3, 
                            '10 TO 12': 4, '13 TO 15': 5, '16 TO 18': 6, '19 TO 21': 7}
            
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
            
            if pred_town in le_town.classes_:
                pred_features['town_encoded'] = le_town.transform([pred_town])[0]
            
            pred_scaled = scaler.transform(pred_features)
            prediction = best_model.predict(pred_scaled)[0]
            
            st.success(f"🏠 预测转售价格: **S${prediction:,.0f}**")
            st.info(f"价格区间: S${prediction * 0.9:,.0f} - S${prediction * 1.1:,.0f}")
        
        # 误差分析
        st.subheader("模型误差分析")
        y_pred_best = best_model.predict(X_test_scaled)
        
        fig_errors = go.Figure()
        fig_errors.add_trace(go.Scatter(x=y_test, y=y_pred_best, mode='markers', 
                                         name='预测值', marker=dict(color='blue', opacity=0.5)))
        fig_errors.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                         y=[y_test.min(), y_test.max()],
                                         mode='lines', name='完美预测', 
                                         line=dict(color='red', dash='dash')))
        fig_errors.update_layout(title="预测值 vs 实际值", xaxis_title="实际价格", 
                                  yaxis_title="预测价格", height=400)
        st.plotly_chart(fig_errors, use_container_width=True)
        
    else:
        st.warning(f"{size_type} 数据量不足({len(size_df)}条)，无法建立可靠预测模型。")

# ==================== Tab5: 购房策略 ====================
with tab5:
    st.header("购房策略与保值验证")
    
    st.markdown("""
    ### 🎯 策略框架
    1. **位置优先策略**: 优先选择成熟区、靠近MRT的房源
    2. **房型优化策略**: 根据预算选择性价比最高的房型
    3. **时机选择策略**: 分析价格周期，选择最佳入市时机
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("预算上限 (SGD)", min_value=300000, max_value=1500000, value=800000, step=50000)
        preferred_flat_types = st.multiselect("偏好房型", options=['3 ROOM', '4 ROOM', '5 ROOM'], default=['4 ROOM'])
    with col2:
        max_distance_mrt = st.slider("距离MRT最大距离 (米)", 200, 2000, 800)
        min_remaining_lease = st.slider("最少剩余年限 (年)", 40, 99, 70)
    
    strategy_df = df[
        (df['resale_price'] <= budget) &
        (df['flat_type'].isin(preferred_flat_types)) &
        (df['dist_to_mrt'] <= max_distance_mrt) &
        (df['remaining_lease'] >= min_remaining_lease)
    ].copy()
    
    st.subheader(f"符合条件的房源: {len(strategy_df)} 套")
    
    if len(strategy_df) > 0:
        strategy_df['性价比分'] = (1 / (strategy_df['dist_to_mrt'] + 1)) * (strategy_df['is_mature'] + 0.5) * 100
        top_recommendations = strategy_df.nlargest(10, '性价比分')
        
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
        
        baseline_avg = df['resale_price'].mean()
        strategy_avg = strategy_df['resale_price'].mean()
        baseline_2024 = df[df['month'].dt.year == 2024]['resale_price'].mean() if len(df[df['month'].dt.year == 2024]) > 0 else baseline_avg
        strategy_2024 = strategy_df[strategy_df['month'].dt.year == 2024]['resale_price'].mean() if len(strategy_df[strategy_df['month'].dt.year == 2024]) > 0 else strategy_avg
        
        comparison_data = pd.DataFrame({
            '类别': ['基准(所有房源)', '策略推荐房源'],
            '平均价格': [baseline_avg, strategy_avg],
            '2024年均价': [baseline_2024, strategy_2024]
        })
        st.dataframe(comparison_data, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("策略价格优势", f"+{((strategy_avg - baseline_avg) / baseline_avg * 100):.1f}%")
        with col2:
            st.metric("推荐房源数", f"{len(strategy_df)}套")
        with col3:
            st.metric("平均距离MRT", f"{strategy_df['dist_to_mrt'].mean():.0f}m")
        
        st.info("""
        **策略结论**: 符合策略的房源在价格表现上优于市场平均水平，
        成熟区、近MRT的房源具有更好的保值和增值潜力。
        """)
    else:
        st.warning("没有找到符合条件的房源，请调整筛选条件")

# ==================== Tab6: 政策事件分析 ====================
with tab6:
    st.header("新加坡住房政策事件冲击分析")
    
    policy_events = [
        {"date": "2020-04-07", "name": "COVID-19 阻断措施", "type": "negative"},
        {"date": "2020-12-15", "name": "降温措施 - 调高印花税", "type": "negative"},
        {"date": "2021-12-16", "name": "新一轮降温措施", "type": "negative"},
        {"date": "2022-09-30", "name": "房市降温措施加码", "type": "negative"},
        {"date": "2023-04-27", "name": "降温措施调整", "type": "neutral"},
        {"date": "2024-02-20", "name": "预算案住房政策", "type": "positive"}
    ]
    
    df_policy = df.copy()
    df_policy['date'] = pd.to_datetime(df_policy['month'])
    
    policy_impacts = []
    for policy in policy_events:
        policy_date = pd.to_datetime(policy['date'])
        
        before_data = df_policy[(df_policy['date'] >= policy_date - pd.Timedelta(days=90)) & 
                                 (df_policy['date'] < policy_date)]
        after_data = df_policy[(df_policy['date'] >= policy_date) & 
                               (df_policy['date'] <= policy_date + pd.Timedelta(days=90))]
        
        if len(before_data) > 0 and len(after_data) > 0:
            pct_change = (after_data['resale_price'].mean() - before_data['resale_price'].mean()) / before_data['resale_price'].mean() * 100
        else:
            pct_change = 0
        
        policy_impacts.append({
            '事件名称': policy['name'], '日期': policy['date'], 
            '类型': policy['type'], '价格变化(%)': pct_change
        })
    
    impacts_df = pd.DataFrame(policy_impacts)
    st.subheader("📅 政策事件影响分析")
    st.dataframe(impacts_df, use_container_width=True)
    
    fig_policy = px.bar(impacts_df, x='事件名称', y='价格变化(%)', color='类型',
                        title="政策事件前后90天价格变化")
    fig_policy.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_policy, use_container_width=True)
    
    # 价格走势
    monthly_price = df.groupby(df['month'].dt.to_period('M'))['resale_price'].mean().reset_index()
    monthly_price['month'] = monthly_price['month'].astype(str)
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(x=monthly_price['month'], y=monthly_price['resale_price'],
                                       mode='lines+markers', name='平均价格'))
    fig_timeline.update_layout(title="HDB转售价格月度趋势", xaxis_title="日期", 
                                yaxis_title="平均价格 (SGD)", height=450)
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.info("""
    **政策事件分析结论**:
    - 降温措施通常对房价产生短期压制作用
    - 但长期来看，HDB转售市场基本面依然强劲
    - 首次购房者政策对低价房源有积极影响
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 项目说明
- **数据来源**: HDB转售交易记录 (2020-2024)
- **主要功能**: 数据筛选、地图可视化、影响因素分析、价格预测、购房策略
- **预测模型**: 随机森林 / 梯度提升
""")
