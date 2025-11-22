# 交通流分析系统

一个基于计算机视觉的交通流分析系统，能够从交通监控视频中提取车辆轨迹，进行分车道交通流参数计算，并生成专业的交通流分析图表。

## 项目概述

本项目提供了一套完整的交通流分析解决方案，包括车辆检测跟踪、轨迹提取、坐标转换、数据清洗和交通流参数分析等功能。系统支持分车道分析，能够生成交通流基本图、时空轨迹图等专业图表。

## 主要功能

### 🚗 车辆轨迹提取
- 基于YOLO模型的车辆检测与跟踪
- 支持多种车辆类型（小汽车、公交车、卡车、摩托车）
- 实时轨迹数据记录与保存

### 🛣️ 分车道分析
- 自动车道分配与标准化
- 分车道流量、速度、密度计算
- 车头时距与车头间距分析

### 📊 交通流参数计算
- 流量-密度关系（q-k图）
- 速度-密度关系（v-k图）  
- 流量-速度关系（q-v图）
- 支持格林希尔治模型、安德伍德模型等理论拟合

### 📈 数据可视化
- 时空轨迹图
- 交通流基本图（带拟合曲线）
- 车头时距/间距分布图
- 专业中文图表输出

## 项目结构

```
traffic_analysis/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── main.py                   # 主程序入口
│   ├── trajectory_extraction.py  # 轨迹提取模块
│   ├── coordinate_transformation.py  # 坐标转换模块
│   ├── clean_trajectories.py     # 数据清洗模块
│   ├── lane_based_analysis.py    # 分车道分析模块
│   ├── plot_spacetime_trajectories.py  # 时空轨迹图绘制
│   ├── add_lane_column.py        # 车道列添加工具
│   └── config.py                 # 配置文件
├── data/                         # 数据目录
│   ├── raw_videos/               # 原始视频文件
│   └── processed/                # 处理后的数据
├── model/                        # 模型文件
│   └── yolov8n.pt               # YOLO模型权重
├── outputs/                      # 输出结果
└── requirements.txt             # 依赖包列表
```

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU加速推荐)

### 安装步骤

1. 克隆项目到本地
```bash
git clone <repository-url>
cd traffic_analysis
```

2. 创建并激活虚拟环境
```bash
conda create -n traffic_analysis python=3.8
conda activate traffic_analysis
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

4. 下载YOLO模型权重
```bash
# 将YOLO模型文件放置在model目录下
mkdir -p model
# 下载yolov8n.pt或使用自己的预训练模型
```

### 基本使用方法

1. **轨迹提取**
```bash
python src/trajectory_extraction.py
```

2. **数据清洗**
```bash
python src/clean_trajectories.py
```

3. **分车道分析**
```bash
python src/lane_based_analysis.py
```

4. **生成时空轨迹图**
```bash
python src/plot_spacetime_trajectories.py
```

### 高级配置

修改 `src/config.py` 文件来调整系统参数：

```python
# 坐标转换参考点
REFERENCE_POINTS = {
    'pixel': [(240,420), (607,420), (356,204), (471,204)],
    'world': [(0,50), (11,50), (0,0), (11,0)]
}

# 道路参数
ROAD_CONFIG = {
    'width': 11,      # 道路宽度（米）
    'length': 50,     # 道路长度（米）
    'num_lanes': 3,   # 车道数量
}

# 分析参数
ANALYSIS_CONFIG = {
    'time_interval': 300,     # 分析时间间隔（秒）
    'max_speed': 50,          # 最大合理速度（m/s）
}
```

## 模块说明

### 1. 轨迹提取模块 (`trajectory_extraction.py`)
- 使用YOLO模型进行车辆检测和跟踪
- 支持多种处理模式（快速、平衡、标准）
- 生成带标注的视频文件

### 2. 坐标转换模块 (`coordinate_transformation.py`)
- 透视变换将像素坐标转换为世界坐标
- 车道分配和行驶方向判断
- 数据清洗和异常值过滤

### 3. 数据清洗模块 (`clean_trajectories.py`)
- 异常轨迹点过滤
- 固定时间步长重采样
- 卡尔曼滤波平滑处理

### 4. 分车道分析模块 (`lane_based_analysis.py`)
- 分车道交通流参数计算
- 车头时距和车头间距分析
- 交通流基本图绘制

### 5. 时空轨迹图模块 (`plot_spacetime_trajectories.py`)
- 生成时空轨迹图
- 支持分车道可视化
- 数据平滑和滤波处理

## 输出结果

系统将生成以下分析结果：

### 数据文件
- `trajectories_cleaned_video1/2.csv` - 清洗后的轨迹数据
- `lane_flow_parameters.csv` - 分车道交通流参数

### 图表文件
- `lane_fundamental_diagram.png` - 交通流基本图
- `headway_distributions.png` - 车头时距/间距分布
- `spacetime_trajectories.png` - 时空轨迹图

## 技术特点

- 🎯 **高精度检测**: 基于YOLO的先进目标检测算法
- 📐 **坐标转换**: 准确的像素到世界坐标转换
- 🧹 **数据清洗**: 多重滤波和异常值处理
- 🛣️ **分车道分析**: 精细化的车道级交通流分析
- 📊 **专业图表**: 符合交通工程标准的可视化
- 🇨🇳 **中文支持**: 完整的中文界面和图表

## 依赖包

主要依赖包包括：
- `ultralytics` - YOLO模型
- `opencv-python` - 图像处理和视频分析
- `pandas`, `numpy` - 数据处理
- `matplotlib`, `seaborn` - 数据可视化
- `scipy`, `scikit-learn` - 科学计算和机器学习

完整依赖列表请参考 `requirements.txt`。

## 故障排除

### 常见问题

1. **权限错误**
   - 确保对数据目录有读写权限
   - 检查文件路径配置

2. **中文字体显示问题**
   - 系统会自动检测可用中文字体
   - 如仍有问题，可手动安装中文字体包

3. **模型加载失败**
   - 检查YOLO模型文件路径
   - 确保模型文件完整

4. **内存不足**
   - 对于长视频，使用快速处理模式
   - 增加系统虚拟内存
