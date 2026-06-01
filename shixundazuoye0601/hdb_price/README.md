# 🏠 新加坡 HDB 组屋转售价格分析与预测系统

基于 Streamlit 的交互式 HDB 组屋转售价格分析与预测 Web 应用。

## 项目概述

使用新加坡政府公开的 HDB 转售成交数据（data.gov.sg），通过特征工程与机器学习，为不同类型的组屋评估价格预测效果，分析各类房源的定价规律，并提供实用的价格预估功能。

## 功能模块

| 模块 | 内容 |
|------|------|
| 📊 数据总览 | 数据筛选、统计看板、成交记录展示 |
| 🗺️ 地图可视化 | 镇区均价地图、热力图、时空变化、配套叠加 |
| 📈 影响因素分析 | 面积/房型/楼层/租约/成熟区/MRT 距离等 7+ 因素分析 |
| 🤖 价格预测 | 多模型对比、分房型评估、特征重要性、价格预估器 |
| 💡 分析思考 | 保值分析、策略逻辑、模型误差三大思考题 |
| 🏆 策略验证 | 购房策略设定、验证期表现、策略组 vs 基准组对比 |
| 📰 政策事件分析 | 事件时间线、前后价格对比、分镇区影响（可选加分模块）|

## 项目结构

```
hdb_price/
├── app.py                  # Streamlit 主程序
├── data/
│   ├── hdb_resale.csv       # HDB 转售成交数据（需自行下载）
│   ├── town_locations.csv   # 镇区中心点坐标
│   ├── mrt_stations.csv     # MRT 站点数据
│   ├── schools.csv          # 学校数据
│   └── events.csv           # 政策事件时间线
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载与清洗
│   ├── geo_utils.py         # 地理工具函数
│   ├── feature_eng.py       # 特征工程
│   ├── model.py             # 模型训练与预测
│   └── analysis.py          # 分析函数
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 HDB 转售数据 CSV 文件放入 `data/` 目录，命名为 `hdb_resale.csv`。

数据下载地址：https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view

也可通过 API 获取：
```python
import requests
import pandas as pd

url = "https://data.gov.sg/api/action/datastore_search"
params = {
    "resource_id": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "limit": 10000,
}
resp = requests.get(url, params=params).json()
df = pd.DataFrame(resp["result"]["records"])
```

### 3. 运行应用

```bash
cd hdb_price
streamlit run app.py
```

应用将在浏览器中自动打开（默认 http://localhost:8501）。

## 数据说明

| 文件 | 说明 | 来源 |
|------|------|------|
| hdb_resale.csv | HDB 转售成交记录（2020年至今）| data.gov.sg |
| town_locations.csv | 26个HDB镇区中心点坐标 | 手工整理 |
| mrt_stations.csv | MRT/LRT 站点坐标 | data.gov.sg / LTA DataMall |
| schools.csv | 学校信息及坐标 | data.gov.sg / MOE |
| events.csv | 政策事件时间线 | 新闻检索 / HDB官网 |

## 技术栈

- **前端框架**: Streamlit
- **数据处理**: Pandas, NumPy
- **可视化**: Plotly, PyDeck
- **机器学习**: Scikit-learn (LinearRegression, Ridge, RandomForest, GradientBoosting)

## 免责声明

本项目仅供学习使用，房价预测结果不构成任何购房建议。
