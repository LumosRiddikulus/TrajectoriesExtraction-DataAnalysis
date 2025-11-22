# 配置文件 - 根据你的实际情况调整这些参数

# 坐标转换参考点配置
# 根据你提供的854×480分辨率下的坐标点
REFERENCE_POINTS = {
    'pixel': [
        (240, 420),   # 左下 - 车辆行驶终点
        (607, 420),   # 右下 - 车辆行驶终点  
        (356, 204),   # 左上 - 车辆行驶起点
        (471, 204)    # 右上 - 车辆行驶起点
    ],
    'world': [
        (0, 50),      # 左下 - 终点位置 (0,50)
        (11, 50),     # 右下 - 终点位置 (11,50)
        (0, 0),       # 左上 - 起点位置 (0,0)
        (11, 0)       # 右上 - 起点位置 (11,0)
    ]
}

# 道路参数
ROAD_CONFIG = {
    'width': 11,      # 道路宽度（米）
    'length': 50,     # 道路长度（米）
    'num_lanes': 3,   # 车道数量
    'direction': 'downhill'  # 行驶方向：downhill(下行), uphill(上行)
}

# 轨迹提取参数
TRACKING_CONFIG = {
    'model_path': '../model/yolov8n.pt',
    'vehicle_classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck
    'conf_threshold': 0.05,           # 检测置信度阈值
    'skip_frames': 1,                 # 跳帧数
    'iou_threshold': 0.4              # IoU阈值
}

# 分析参数
ANALYSIS_CONFIG = {
    'time_interval': 300,             # 分析时间间隔（秒）
    'min_trajectory_points': 5,       # 最小轨迹点数
    'max_speed': 50,                  # 最大合理速度（m/s）
    'direction_threshold': 0.5        # 方向判断阈值（m/s）
}

# 文件路径配置
PATH_CONFIG = {
    'raw_video': '../data/raw_videos/traffic_video.mp4',
    'raw_trajectories': '../data/processed/trajectories_raw.csv',
    'processed_trajectories': '../data/processed/trajectories_processed.csv',
    'enhanced_trajectories': '../data/processed/trajectories_enhanced.csv',
    'flow_parameters': '../data/processed/flow_parameters.csv',
    'output_dir': '../data/outputs/'
}