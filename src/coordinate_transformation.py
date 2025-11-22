import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import REFERENCE_POINTS, ROAD_CONFIG, ANALYSIS_CONFIG

class CoordinateTransformer:
    def __init__(self, reference_points=None):
        """
        坐标转换器
        reference_points: {
            'pixel': [(x1,y1), (x2,y2), ...],
            'world': [(X1,Y1), (X2,Y2), ...]
        }
        """
        self.reference_points = reference_points or REFERENCE_POINTS
        self.pixel_points = np.float32(self.reference_points['pixel'])
        self.world_points = np.float32(self.reference_points['world'])
        
        # 计算透视变换矩阵
        self.transform_matrix = cv2.getPerspectiveTransform(
            self.pixel_points, 
            self.world_points
        )
        
        # 验证变换矩阵
        self._validate_transformation()
    
    def _validate_transformation(self):
        """验证变换矩阵的准确性"""
        print("🔍 验证坐标转换矩阵...")
        
        # 将参考像素点转换到世界坐标
        transformed = cv2.perspectiveTransform(
            self.pixel_points.reshape(-1, 1, 2), 
            self.transform_matrix
        ).reshape(-1, 2)
        
        # 计算转换误差
        errors = np.linalg.norm(transformed - self.world_points, axis=1)
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"✅ 坐标转换验证完成:")
        print(f"   平均误差: {avg_error:.3f} 米")
        print(f"   最大误差: {max_error:.3f} 米")
        
        if avg_error > 1.0:
            print("⚠️ 警告: 坐标转换误差较大，请检查参考点设置")
    
    def pixel_to_world(self, pixel_points):
        """
        将像素坐标转换为世界坐标
        pixel_points: numpy数组，形状为 (N, 2)
        """
        if len(pixel_points) == 0:
            return np.array([])
            
        # 确保输入格式正确
        pixel_points = np.float32(pixel_points).reshape(-1, 1, 2)
        
        # 应用透视变换
        world_points = cv2.perspectiveTransform(pixel_points, self.transform_matrix)
        
        return world_points.reshape(-1, 2)

class LaneBasedAnalyzer:
    def __init__(self, trajectories_df, road_width=None, num_lanes=None):
        self.df = trajectories_df
        self.road_width = road_width or ROAD_CONFIG['width']
        self.num_lanes = num_lanes or ROAD_CONFIG['num_lanes']
        self.lane_width = self.road_width / self.num_lanes
        
    def assign_lanes(self, reference_line='left'):
        """
        根据横向位置分配车道
        reference_line: 'left', 'right', 'center'
        """
        print(f"🛣️  开始车道分配: {self.num_lanes}车道, 道路宽度{self.road_width}米")
        
        if self.df.empty:
            print("❌ 没有轨迹数据")
            return self.df
        
        # 计算车辆的横向位置
        if 'y_world' not in self.df.columns:
            print("❌ 缺少横向坐标数据(y_world)")
            return self.df
        
        # 确定参考线位置
        if reference_line == 'left':
            ref_position = self.df['y_world'].min()
        elif reference_line == 'right':
            ref_position = self.df['y_world'].max()
        else:  # center
            ref_position = (self.df['y_world'].min() + self.df['y_world'].max()) / 2
        
        # 计算相对位置
        self.df['lateral_position'] = self.df['y_world'] - ref_position
        
        # 分配车道
        lane_boundaries = np.linspace(0, self.road_width, self.num_lanes + 1)
        
        def get_lane_number(lateral_pos):
            for i in range(self.num_lanes):
                if lane_boundaries[i] <= lateral_pos < lane_boundaries[i+1]:
                    return i + 1
            return 0  # 超出道路范围
        
        self.df['lane'] = self.df['lateral_position'].apply(get_lane_number)
        
        # 统计车道分配结果
        lane_counts = self.df['lane'].value_counts().sort_index()
        print("📊 车道分配结果:")
        for lane, count in lane_counts.items():
            if lane > 0:
                print(f"  车道{lane}: {count}个轨迹点")
            else:
                print(f"  超出道路: {count}个轨迹点")
        
        return self.df
    
    def separate_directions(self, direction_threshold=None):
        """
        分离行驶方向
        """
        direction_threshold = direction_threshold or ANALYSIS_CONFIG['direction_threshold']
        print("🔄 分离行驶方向...")
        
        if self.df.empty:
            return self.df
        
        # 为每个车辆计算平均速度方向
        vehicle_directions = {}
        
        for track_id in self.df['track_id'].unique():
            vehicle_data = self.df[self.df['track_id'] == track_id].sort_values('time')
            
            if len(vehicle_data) < 2:
                continue
                
            # 计算主要行驶方向
            if 'speed' not in vehicle_data.columns:
                # 计算速度
                dx = np.diff(vehicle_data['x_world'])
                dt = np.diff(vehicle_data['time'])
                valid_mask = dt > 0
                if valid_mask.any():
                    speeds = dx[valid_mask] / dt[valid_mask]
                    avg_speed = np.mean(speeds)
                else:
                    avg_speed = 0
            else:
                avg_speed = vehicle_data['speed'].mean()
            
            # 根据平均速度符号判断方向
            if avg_speed > direction_threshold:
                direction = 'forward'
            elif avg_speed < -direction_threshold:
                direction = 'backward'
            else:
                direction = 'stationary'
            
            vehicle_directions[track_id] = direction
        
        # 分配方向标签
        self.df['direction'] = self.df['track_id'].map(vehicle_directions).fillna('unknown')
        
        # 统计方向分布
        direction_counts = self.df['direction'].value_counts()
        print("📊 行驶方向分布:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count}个轨迹点")
        
        return self.df
    
    def calculate_mileage_from_start(self, start_position=None):
        """
        计算距离起点的里程
        """
        print("📏 计算里程...")
        
        if self.df.empty:
            return self.df
        
        # 如果没有指定起点，使用最小位置作为起点
        if start_position is None:
            start_position = self.df['x_world'].min()
        
        # 计算里程（距离起点的距离）
        self.df['mileage'] = self.df['x_world'] - start_position
        
        print(f"  起点位置: {start_position:.1f}米")
        print(f"  里程范围: {self.df['mileage'].min():.1f} - {self.df['mileage'].max():.1f}米")
        
        return self.df

def clean_trajectory_data(trajectories_df, min_points=None, max_speed=None):
    """
    清洗轨迹数据
    """
    min_points = min_points or ANALYSIS_CONFIG['min_trajectory_points']
    max_speed = max_speed or ANALYSIS_CONFIG['max_speed']
    
    print("🧹 开始清洗轨迹数据...")
    
    if trajectories_df.empty:
        return trajectories_df
    
    original_count = len(trajectories_df)
    original_vehicles = trajectories_df['track_id'].nunique()
    
    # 1. 过滤轨迹点过少的车辆
    points_per_vehicle = trajectories_df.groupby('track_id').size()
    valid_vehicles = points_per_vehicle[points_per_vehicle >= min_points].index
    trajectories_df = trajectories_df[trajectories_df['track_id'].isin(valid_vehicles)]
    
    print(f"   过滤短轨迹: {original_vehicles} → {trajectories_df['track_id'].nunique()} 辆车")
    
    # 2. 计算速度并过滤异常值
    trajectories_df = trajectories_df.sort_values(['track_id', 'time'])
    
    # 计算速度
    trajectories_df['dx'] = trajectories_df.groupby('track_id')['x_world'].diff()
    trajectories_df['dy'] = trajectories_df.groupby('track_id')['y_world'].diff()
    trajectories_df['dt'] = trajectories_df.groupby('track_id')['time'].diff()
    
    # 避免除零
    valid_dt = trajectories_df['dt'] > 0
    trajectories_df.loc[valid_dt, 'speed'] = (
        np.sqrt(trajectories_df['dx']**2 + trajectories_df['dy']**2) / 
        trajectories_df['dt']
    ).fillna(0)
    
    # 过滤异常速度
    speed_mask = (trajectories_df['speed'] <= max_speed) & (trajectories_df['speed'] >= 0)
    trajectories_df = trajectories_df[speed_mask]
    
    print(f"   过滤异常速度: {original_count} → {len(trajectories_df)} 个轨迹点")
    
    # 移除临时列
    trajectories_df = trajectories_df.drop(['dx', 'dy', 'dt'], axis=1)
    
    return trajectories_df

def process_trajectory_data(trajectories_df, transformer):
    """
    完整的轨迹数据处理流程
    """
    print("🔄 开始轨迹数据处理...")
    
    if trajectories_df.empty:
        print("⚠️ 轨迹数据为空，跳过处理")
        return trajectories_df
    
    # 1. 坐标转换
    pixel_coords = trajectories_df[['x', 'y']].values
    world_coords = transformer.pixel_to_world(pixel_coords)
    
    trajectories_df['x_world'] = world_coords[:, 0]
    trajectories_df['y_world'] = world_coords[:, 1]
    
    print(f"✅ 坐标转换完成: {len(trajectories_df)} 个点")
    
    # 2. 数据清洗
    trajectories_df = clean_trajectory_data(trajectories_df)
    
    # 3. 车道和方向分析
    analyzer = LaneBasedAnalyzer(trajectories_df)
    trajectories_df = analyzer.assign_lanes()
    trajectories_df = analyzer.separate_directions()
    trajectories_df = analyzer.calculate_mileage_from_start()
    
    print(f"🎯 最终数据: {trajectories_df['track_id'].nunique()} 辆车, {len(trajectories_df)} 个轨迹点")
    
    return trajectories_df