"""
分车道交通流分析脚本
基于两个CSV文件进行分车道的流量、速度、密度分析
计算分车道的车头时距和车头间距，并绘制交通流基本图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
from scipy import stats
from tqdm import tqdm

# 尝试导入seaborn（可选）
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import ROAD_CONFIG, ANALYSIS_CONFIG
from coordinate_transformation import CoordinateTransformer

# 设置中文字体 - 自动检测并使用可用的中文字体
def setup_chinese_font():
    """设置中文字体，优先使用系统可用的中文字体"""
    # 常见的中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimHei',                    # 黑体（Windows）
        'Microsoft YaHei',           # 微软雅黑（Windows）
        'WenQuanYi Micro Hei',       # 文泉驿微米黑（Linux）
        'WenQuanYi Zen Hei',        # 文泉驿正黑（Linux）
        'Noto Sans CJK SC',          # Noto Sans（Linux/通用）
        'Noto Sans CJK JP',          # Noto Sans（Linux/通用）
        'Source Han Sans CN',        # 思源黑体（Linux）
        'STHeiti',                   # 华文黑体（macOS）
        'Arial Unicode MS',          # Arial Unicode（通用）
        'DejaVu Sans'                # 备用字体
    ]
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print(f"✅ 已设置中文字体: {selected_font}")
    else:
        # 如果没有找到常见字体，尝试查找包含CJK或中文的字体
        cjk_fonts = [f for f in available_fonts if 'CJK' in f or 'Chinese' in f or 'SC' in f or 'CN' in f]
        if cjk_fonts:
            selected_font = cjk_fonts[0]
            plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
            print(f"✅ 已设置中文字体: {selected_font}")
        else:
            print("⚠️ 警告: 未找到中文字体，中文可能显示为方块")
            # 使用默认字体，但设置unicode_minus
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()


class LaneBasedTrafficAnalyzer:
    """分车道交通流分析器"""
    
    def __init__(self, trajectories_df, road_length=None, time_interval=None):
        """
        初始化分析器
        
        Args:
            trajectories_df: 轨迹数据DataFrame
            road_length: 道路长度（米）
            time_interval: 分析时间间隔（秒），默认300秒（5分钟）
        """
        self.df = trajectories_df.copy()
        self.road_length = road_length or ROAD_CONFIG['length']
        self.time_interval = time_interval or ANALYSIS_CONFIG['time_interval']
        
        # 预处理数据
        self._preprocess_data()
        
        print(f"📊 分车道分析器初始化完成")
        print(f"   道路长度: {self.road_length} 米")
        print(f"   时间间隔: {self.time_interval} 秒 ({self.time_interval/60:.1f} 分钟)")
        print(f"   数据行数: {len(self.df)}")
        print(f"   唯一车辆数: {self.df['track_id'].nunique()}")
    
    def _preprocess_data(self):
        """数据预处理：坐标转换、速度计算、车道标准化"""
        
        print("\n🔄 开始数据预处理...")
        
        # 1. 检查是否需要坐标转换
        if 'x_world' not in self.df.columns or 'y_world' not in self.df.columns:
            print("   ⚠️ 缺少世界坐标，进行坐标转换...")
            transformer = CoordinateTransformer()
            pixel_coords = self.df[['x', 'y']].values
            world_coords = transformer.pixel_to_world(pixel_coords)
            self.df['x_world'] = world_coords[:, 0]
            self.df['y_world'] = world_coords[:, 1]
        
        # 2. 计算速度（如果不存在）
        if 'speed' not in self.df.columns:
            print("   ⚠️ 缺少速度数据，进行计算...")
            self.df = self._calculate_speed()
        
        # 3. 标准化车道编号
        if 'lane' in self.df.columns:
            print("   🔄 标准化车道编号...")
            self.df['lane'] = self.df['lane'].astype(str)
            # 将车道字符串转换为数字
            lane_mapping = {
                'lan1': 1, 'lane1': 1, '1': 1,
                'lan2': 2, 'lane2': 2, '2': 2,
                'lan3': 3, 'lane3': 3, '3': 3,
                'non-motor': 0, 'non_motor': 0, 'nonmotor': 0
            }
            self.df['lane_num'] = self.df['lane'].map(lane_mapping).fillna(0).astype(int)
        else:
            print("   ⚠️ 缺少车道数据，无法进行分车道分析")
            self.df['lane_num'] = 0
        
        # 4. 过滤异常数据
        if 'speed' in self.df.columns:
            max_speed = ANALYSIS_CONFIG.get('max_speed', 50)
            before_speed_filter = len(self.df)
            self.df = self.df[self.df['speed'] <= max_speed]
            self.df = self.df[self.df['speed'] >= 0]
            if before_speed_filter != len(self.df):
                print(f"   ⚠️ 速度过滤: {before_speed_filter} -> {len(self.df)} 行")
        
        # 5. 过滤超出道路范围的x_world数据（可选，根据实际情况调整）
        if 'x_world' in self.df.columns:
            # 允许一定的容差范围（例如道路长度的2倍）
            x_tolerance = self.road_length * 2
            before_x_filter = len(self.df)
            # 只过滤明显异常的数据（例如超出道路长度10倍的数据）
            self.df = self.df[
                (self.df['x_world'] >= -self.road_length * 10) & 
                (self.df['x_world'] <= self.road_length * 10)
            ]
            if before_x_filter != len(self.df):
                print(f"   ⚠️ x_world范围过滤: {before_x_filter} -> {len(self.df)} 行")
        
        # 6. 输出车道数据统计
        if 'lane_num' in self.df.columns:
            print(f"\n   📊 各车道数据统计:")
            lane_counts = self.df['lane_num'].value_counts().sort_index()
            for lane_num, count in lane_counts.items():
                if lane_num > 0:
                    unique_vehicles = self.df[self.df['lane_num'] == lane_num]['track_id'].nunique()
                    print(f"      车道 {lane_num}: {count} 行, {unique_vehicles} 辆唯一车辆")
        
        print(f"   ✅ 预处理完成: {len(self.df)} 个有效数据点")
    
    def _calculate_speed(self):
        """计算车辆速度"""
        df = self.df.copy()
        df = df.sort_values(['track_id', 'time'])
        
        # 计算位置差和时间差
        df['dx'] = df.groupby('track_id')['x_world'].diff()
        df['dy'] = df.groupby('track_id')['y_world'].diff()
        df['dt'] = df.groupby('track_id')['time'].diff()
        
        # 计算速度
        df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['speed'] = df['distance'] / df['dt']
        df['speed'] = df['speed'].fillna(0)
        
        # 过滤异常速度值
        df.loc[df['speed'] > 50, 'speed'] = 0
        df.loc[df['speed'] < 0, 'speed'] = 0
        
        return df
    
    def calculate_lane_flow_parameters(self):
        """
        分车道按5分钟时间段计算流量、速度、密度
        
        Returns:
            DataFrame: 包含每个时间段每个车道的流量、速度、密度
        """
        print("\n📊 开始计算分车道交通流参数...")
        
        if 'lane_num' not in self.df.columns:
            print("❌ 缺少车道数据，无法进行分车道分析")
            return pd.DataFrame()
        
        # 确定分析时间段
        total_time = self.df['time'].max()
        time_intervals = np.arange(0, total_time + self.time_interval, self.time_interval)
        
        print(f"   总时长: {total_time/60:.1f} 分钟")
        print(f"   时间段数: {len(time_intervals)-1}")
        
        flow_data = []
        lanes = sorted([l for l in self.df['lane_num'].unique() if l > 0])
        print(f"   分析车道: {lanes}")
        
        for i in tqdm(range(len(time_intervals) - 1), desc="计算时间段"):
            start_time = time_intervals[i]
            end_time = time_intervals[i + 1]
            
            # 筛选时间段内的数据
            interval_data = self.df[
                (self.df['time'] >= start_time) & 
                (self.df['time'] < end_time)
            ]
            
            if len(interval_data) == 0:
                continue
            
            # 对每个车道计算参数
            for lane in lanes:
                lane_data = interval_data[interval_data['lane_num'] == lane]
                
                if len(lane_data) == 0:
                    continue
                
                # 计算流量（辆/小时）
                unique_vehicles = lane_data['track_id'].nunique()
                interval_duration = end_time - start_time
                flow_rate = (unique_vehicles / interval_duration) * 3600 if interval_duration > 0 else 0
                
                # 计算时间平均速度（米/秒）
                if 'speed' in lane_data.columns and len(lane_data[lane_data['speed'] > 0]) > 0:
                    time_mean_speed = lane_data[lane_data['speed'] > 0]['speed'].mean()
                else:
                    time_mean_speed = 0
                
                # 计算密度（辆/公里）- 使用基本方程 k = q / v
                if time_mean_speed > 0:
                    speed_kmh = time_mean_speed * 3.6  # m/s 转 km/h
                    density = (flow_rate / speed_kmh) * 1000 if speed_kmh > 0 else 0  # veh/km
                else:
                    # 如果速度为零，使用直接测量法
                    time_samples = np.linspace(start_time, end_time, min(10, int(end_time - start_time)))
                    vehicle_counts = []
                    for t in time_samples:
                        vehicles_at_t = lane_data[
                            (lane_data['time'] >= t - 1) & (lane_data['time'] <= t + 1)
                        ]['track_id'].nunique()
                        vehicle_counts.append(vehicles_at_t)
                    avg_vehicle_count = np.mean(vehicle_counts) if vehicle_counts else 0
                    density = (avg_vehicle_count / self.road_length) * 1000
                
                flow_data.append({
                    'time_interval': f"{start_time/60:.1f}-{end_time/60:.1f}min",
                    'start_time': start_time,
                    'end_time': end_time,
                    'lane': lane,
                    'flow': flow_rate,
                    'time_mean_speed': time_mean_speed,
                    'density': density,
                    'vehicle_count': unique_vehicles,
                    'interval_duration': interval_duration
                })
        
        result_df = pd.DataFrame(flow_data)
        
        print(f"\n✅ 分车道交通流参数计算完成: {len(result_df)} 条记录")
        if not result_df.empty:
            print(f"   流量范围: {result_df['flow'].min():.1f} - {result_df['flow'].max():.1f} 辆/小时")
            print(f"   密度范围: {result_df['density'].min():.1f} - {result_df['density'].max():.1f} 辆/公里")
            print(f"   速度范围: {result_df['time_mean_speed'].min():.1f} - {result_df['time_mean_speed'].max():.1f} m/s")
        
        return result_df
    
    def calculate_lane_headway_distribution(self, observation_section=None, use_multiple_sections=True):
        """
        分车道计算车头时距分布
        
        Args:
            observation_section: 观测断面位置（米），默认道路中点
            use_multiple_sections: 是否使用多个观测断面以提高数据利用率
        
        Returns:
            dict: {lane: headway_array}
        """
        print("\n📊 开始计算分车道车头时距...")
        
        lanes = sorted([l for l in self.df['lane_num'].unique() if l > 0])
        headway_results = {}
        
        for lane in lanes:
            print(f"\n   处理车道 {lane}...")
            lane_data = self.df[self.df['lane_num'] == lane].copy()
            
            if len(lane_data) < 2:
                print(f"      ⚠️ 车道 {lane} 数据不足")
                headway_results[lane] = np.array([])
                continue
            
            # 确定观测断面位置
            if use_multiple_sections:
                # 使用多个观测断面以提高数据利用率
                # 在道路的1/4、1/2、3/4位置设置观测断面
                observation_sections = [
                    self.road_length * 0.25,
                    self.road_length * 0.5,
                    self.road_length * 0.75
                ]
                print(f"      使用多个观测断面: {[f'{s:.1f}' for s in observation_sections]} 米")
            else:
                if observation_section is None:
                    observation_section = self.road_length / 2
                observation_sections = [observation_section]
                print(f"      观测断面位置: {observation_section:.1f} 米")
            
            # 为每个车辆找到通过任一观测断面的时间
            crossing_times = {}  # {track_id: [(section, time), ...]}
            
            for track_id in lane_data['track_id'].unique():
                vehicle_data = lane_data[lane_data['track_id'] == track_id].sort_values('time')
                
                if len(vehicle_data) < 2:
                    continue
                
                x_positions = vehicle_data['x_world'].values
                times = vehicle_data['time'].values
                
                # 检查车辆是否跨越了任一观测断面
                for obs_section in observation_sections:
                    for i in range(len(x_positions) - 1):
                        if (x_positions[i] <= obs_section and x_positions[i + 1] >= obs_section) or \
                           (x_positions[i] >= obs_section and x_positions[i + 1] <= obs_section):
                            
                            # 线性插值计算确切通过时间
                            t1, t2 = times[i], times[i + 1]
                            x1, x2 = x_positions[i], x_positions[i + 1]
                            
                            if x1 != x2:
                                cross_time = t1 + (t2 - t1) * (obs_section - x1) / (x2 - x1)
                                if track_id not in crossing_times:
                                    crossing_times[track_id] = []
                                crossing_times[track_id].append((obs_section, cross_time))
                                break
            
            # 对于每辆车，选择最早通过观测断面的时间
            vehicle_crossing_times = {}
            for track_id, crossings in crossing_times.items():
                if crossings:
                    # 选择最早通过的时间
                    earliest = min(crossings, key=lambda x: x[1])
                    vehicle_crossing_times[track_id] = earliest[1]
            
            if len(vehicle_crossing_times) < 2:
                print(f"      ⚠️ 车道 {lane} 只有 {len(vehicle_crossing_times)} 辆车通过观测断面")
                headway_results[lane] = np.array([])
                continue
            
            # 按通过时间排序
            sorted_times = sorted(vehicle_crossing_times.items(), key=lambda x: x[1])
            
            # 计算车头时距
            headways = []
            for i in range(len(sorted_times) - 1):
                time_gap = sorted_times[i + 1][1] - sorted_times[i][1]
                if 0.1 < time_gap < 60:  # 合理范围
                    headways.append(time_gap)
            
            headway_results[lane] = np.array(headways)
            print(f"      ✅ 车道 {lane}: {len(vehicle_crossing_times)} 辆车通过观测断面, {len(headways)} 个有效时距, 平均={np.mean(headways):.2f}s" if headways else f"      ⚠️ 车道 {lane}: {len(vehicle_crossing_times)} 辆车通过观测断面, 但无有效时距")
        
        return headway_results
    
    def calculate_lane_space_headway_distribution(self, num_samples=100):
        """
        分车道计算车头间距分布
        
        Args:
            num_samples: 采样时间点数量
        
        Returns:
            dict: {lane: space_headway_array}
        """
        print("\n📊 开始计算分车道车头间距...")
        
        lanes = sorted([l for l in self.df['lane_num'].unique() if l > 0])
        space_headway_results = {}
        
        # 车辆长度估算
        vehicle_length_map = {
            'car': 4.5,
            'bus': 12.0,
            'truck': 8.0,
            'motorcycle': 2.0
        }
        
        time_range = (self.df['time'].min(), self.df['time'].max())
        time_samples = np.linspace(time_range[0], time_range[1], num_samples)
        
        for lane in lanes:
            print(f"\n   处理车道 {lane}...")
            lane_data = self.df[self.df['lane_num'] == lane].copy()
            
            if len(lane_data) < 2:
                print(f"      ⚠️ 车道 {lane} 数据不足")
                space_headway_results[lane] = np.array([])
                continue
            
            space_headways = []
            
            for t in tqdm(time_samples, desc=f"  车道 {lane}", leave=False):
                # 找到在时间t位于路段上的车辆
                vehicles_at_t = []
                
                for track_id in lane_data['track_id'].unique():
                    vehicle_data = lane_data[lane_data['track_id'] == track_id].sort_values('time')
                    time_diff = np.abs(vehicle_data['time'] - t)
                    min_idx = time_diff.idxmin()
                    
                    if time_diff[min_idx] < 1.0:  # 1秒容忍度
                        vehicle_type = vehicle_data.loc[min_idx, 'vehicle_type'] if 'vehicle_type' in vehicle_data.columns else 'car'
                        vehicle_length = vehicle_length_map.get(vehicle_type, 4.5)
                        
                        vehicles_at_t.append({
                            'track_id': track_id,
                            'x_world': vehicle_data.loc[min_idx, 'x_world'],
                            'vehicle_length': vehicle_length
                        })
                
                # 按位置排序（从前往后）
                vehicles_at_t.sort(key=lambda x: x['x_world'], reverse=True)
                
                # 计算相邻车辆的车头间距
                for i in range(len(vehicles_at_t) - 1):
                    x_i_minus_1 = vehicles_at_t[i]['x_world']  # 前车位置
                    x_i = vehicles_at_t[i + 1]['x_world']  # 后车位置
                    l_i_minus_1 = vehicles_at_t[i]['vehicle_length']  # 前车长度
                    
                    d_i = x_i_minus_1 - x_i - l_i_minus_1  # 车头间距
                    
                    if 2 < d_i < 200:  # 合理范围（米）
                        space_headways.append(d_i)
            
            space_headway_results[lane] = np.array(space_headways)
            print(f"      ✅ 车道 {lane}: {len(space_headways)} 个有效间距, 平均={np.mean(space_headways):.2f}m" if space_headways else f"      ⚠️ 车道 {lane}: 无有效间距")
        
        return space_headway_results
    
    def plot_lane_fundamental_diagram(self, flow_params_df, output_path=None):
        """
        绘制分车道的交通流基本图，并添加多种拟合曲线
        
        Args:
            flow_params_df: 流量参数DataFrame（包含lane列）
            output_path: 输出图片路径
        """
        print("\n📊 开始绘制分车道交通流基本图（带多种拟合曲线）...")
        
        if flow_params_df.empty or 'lane' not in flow_params_df.columns:
            print("❌ 数据为空或缺少车道信息")
            return
        
        lanes = sorted([l for l in flow_params_df['lane'].unique() if l > 0])
        
        if len(lanes) == 0:
            print("❌ 没有有效的车道数据")
            return
        
        # 设置颜色
        colors = plt.cm.Set1(np.linspace(0, 1, len(lanes)))
        lane_colors = dict(zip(lanes, colors))
        
        # 创建图形：3个子图（q-k图、v-k图、q-v图）
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        
        def greenshields_model(k, vf, kj):
            """格林希尔治模型: q = vf * k * (1 - k/kj)"""
            return vf * k * (1 - k/kj)
        
        def underwood_model(k, vf, k0):
            """安德伍德指数模型: v = vf * exp(-k/k0)"""
            return vf * np.exp(-k/k0)
        
        # 1. 流量-密度图（q-k图）
        ax1 = axes[0]
        for lane in lanes:
            lane_data = flow_params_df[flow_params_df['lane'] == lane]
            if len(lane_data) > 0:
                # 绘制散点图
                scatter = ax1.scatter(lane_data['density'], lane_data['flow'], 
                        label=f'车道 {lane}', color=lane_colors[lane], 
                        alpha=0.7, s=60)
                
                # 添加多种拟合曲线
                if len(lane_data) >= 4:
                    try:
                        sorted_data = lane_data.sort_values('density')
                        x_fit = sorted_data['density'].values
                        y_fit = sorted_data['flow'].values
                        
                        # 过滤掉异常值
                        valid_mask = (x_fit > 0) & (y_fit > 0)
                        x_fit = x_fit[valid_mask]
                        y_fit = y_fit[valid_mask]
                        
                        if len(x_fit) >= 4:
                            # 方法1: 二次多项式拟合
                            coeffs_poly = np.polyfit(x_fit, y_fit, 2)
                            poly = np.poly1d(coeffs_poly)
                            
                            # 方法2: 格林希尔治模型拟合
                            try:
                                from scipy.optimize import curve_fit
                                # 估计初始参数
                                vf_guess = max(y_fit / x_fit) if max(x_fit) > 0 else 20
                                kj_guess = max(x_fit) * 1.2
                                
                                popt, pcov = curve_fit(greenshields_model, x_fit, y_fit, 
                                                    p0=[vf_guess, kj_guess], 
                                                    bounds=([0, max(x_fit)*1.1], [50, max(x_fit)*3]))
                                vf_fit, kj_fit = popt
                                
                                # 生成拟合曲线
                                x_line = np.linspace(0, kj_fit, 100)
                                y_line_green = greenshields_model(x_line, vf_fit, kj_fit)
                                
                                # 绘制格林希尔治拟合曲线
                                ax1.plot(x_line, y_line_green, color=lane_colors[lane], 
                                        linestyle='-', linewidth=2, alpha=0.8,
                                        label=f'车道 {lane} 格林希尔治拟合')
                                
                                # 计算R²值
                                y_pred_green = greenshields_model(x_fit, vf_fit, kj_fit)
                                ss_res_green = np.sum((y_fit - y_pred_green) ** 2)
                                ss_tot_green = np.sum((y_fit - np.mean(y_fit)) ** 2)
                                r_squared_green = 1 - (ss_res_green / ss_tot_green) if ss_tot_green != 0 else 0
                                
                                print(f"   ✅ 车道 {lane} 格林希尔治拟合: vf={vf_fit:.2f}, kj={kj_fit:.2f}, R²={r_squared_green:.3f}")
                                
                            except Exception as e:
                                print(f"   ⚠️ 车道 {lane} 格林希尔治拟合失败: {e}")
                                # 回退到多项式拟合
                                x_line_poly = np.linspace(x_fit.min(), x_fit.max(), 100)
                                y_line_poly = poly(x_line_poly)
                                ax1.plot(x_line_poly, y_line_poly, color=lane_colors[lane], 
                                        linestyle='--', linewidth=2, alpha=0.8,
                                        label=f'车道 {lane} 多项式拟合')
                                
                    except Exception as e:
                        print(f"   ⚠️ 车道 {lane} q-k图拟合失败: {e}")
        
        ax1.set_xlabel('密度 k (veh/km)', fontsize=12)
        ax1.set_ylabel('流量 q (veh/h)', fontsize=12)
        ax1.set_title('流量-密度关系 (q-k图)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 速度-密度图（v-k图）
        ax2 = axes[1]
        for lane in lanes:
            lane_data = flow_params_df[flow_params_df['lane'] == lane]
            if len(lane_data) > 0:
                # 绘制散点图
                ax2.scatter(lane_data['density'], lane_data['time_mean_speed'], 
                        label=f'车道 {lane}', color=lane_colors[lane], 
                        alpha=0.7, s=60)
                
                # 添加拟合曲线
                if len(lane_data) >= 3:
                    try:
                        sorted_data = lane_data.sort_values('density')
                        x_fit = sorted_data['density'].values
                        y_fit = sorted_data['time_mean_speed'].values
                        
                        valid_mask = (x_fit > 0) & (y_fit > 0)
                        x_fit = x_fit[valid_mask]
                        y_fit = y_fit[valid_mask]
                        
                        if len(x_fit) >= 3:
                            # 尝试安德伍德指数模型
                            try:
                                from scipy.optimize import curve_fit
                                vf_guess = max(y_fit)
                                k0_guess = np.mean(x_fit)
                                
                                popt, pcov = curve_fit(underwood_model, x_fit, y_fit, 
                                                    p0=[vf_guess, k0_guess],
                                                    bounds=([0, 0], [50, max(x_fit)*2]))
                                vf_fit, k0_fit = popt
                                
                                x_line = np.linspace(0, max(x_fit)*1.2, 100)
                                y_line_underwood = underwood_model(x_line, vf_fit, k0_fit)
                                
                                ax2.plot(x_line, y_line_underwood, color=lane_colors[lane], 
                                        linestyle='-', linewidth=2, alpha=0.8,
                                        label=f'车道 {lane} 指数拟合')
                                
                                # 计算R²值
                                y_pred_underwood = underwood_model(x_fit, vf_fit, k0_fit)
                                ss_res_underwood = np.sum((y_fit - y_pred_underwood) ** 2)
                                ss_tot_underwood = np.sum((y_fit - np.mean(y_fit)) ** 2)
                                r_squared_underwood = 1 - (ss_res_underwood / ss_tot_underwood) if ss_tot_underwood != 0 else 0
                                
                                print(f"   ✅ 车道 {lane} 安德伍德拟合: vf={vf_fit:.2f}, k0={k0_fit:.2f}, R²={r_squared_underwood:.3f}")
                                
                            except Exception as e:
                                print(f"   ⚠️ 车道 {lane} 指数拟合失败: {e}")
                                # 回退到线性拟合
                                coeffs_linear = np.polyfit(x_fit, y_fit, 1)
                                poly_linear = np.poly1d(coeffs_linear)
                                x_line_linear = np.linspace(x_fit.min(), x_fit.max(), 100)
                                y_line_linear = poly_linear(x_line_linear)
                                ax2.plot(x_line_linear, y_line_linear, color=lane_colors[lane], 
                                        linestyle='--', linewidth=2, alpha=0.8,
                                        label=f'车道 {lane} 线性拟合')
                                
                    except Exception as e:
                        print(f"   ⚠️ 车道 {lane} v-k图拟合失败: {e}")
        
        ax2.set_xlabel('密度 k (veh/km)', fontsize=12)
        ax2.set_ylabel('速度 v (m/s)', fontsize=12)
        ax2.set_title('速度-密度关系 (v-k图)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 流量-速度图（q-v图）
        ax3 = axes[2]
        for lane in lanes:
            lane_data = flow_params_df[flow_params_df['lane'] == lane]
            if len(lane_data) > 0:
                # 绘制散点图
                ax3.scatter(lane_data['time_mean_speed'], lane_data['flow'], 
                        label=f'车道 {lane}', color=lane_colors[lane], 
                        alpha=0.7, s=60)
                
                # 添加拟合曲线
                if len(lane_data) >= 3:
                    try:
                        sorted_data = lane_data.sort_values('time_mean_speed')
                        x_fit = sorted_data['time_mean_speed'].values
                        y_fit = sorted_data['flow'].values
                        
                        valid_mask = (x_fit > 0) & (y_fit > 0)
                        x_fit = x_fit[valid_mask]
                        y_fit = y_fit[valid_mask]
                        
                        if len(x_fit) >= 3:
                            # 使用二次多项式拟合
                            coeffs = np.polyfit(x_fit, y_fit, 2)
                            poly = np.poly1d(coeffs)
                            
                            x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
                            y_line = poly(x_line)
                            
                            ax3.plot(x_line, y_line, color=lane_colors[lane], 
                                    linestyle='--', linewidth=2, alpha=0.8,
                                    label=f'车道 {lane} 多项式拟合')
                            
                            # 计算R²值
                            y_pred = poly(x_fit)
                            ss_res = np.sum((y_fit - y_pred) ** 2)
                            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                            
                            print(f"   ✅ 车道 {lane} q-v图拟合: R² = {r_squared:.3f}")
                            
                    except Exception as e:
                        print(f"   ⚠️ 车道 {lane} q-v图拟合失败: {e}")
        
        ax3.set_xlabel('速度 v (m/s)', fontsize=12)
        ax3.set_ylabel('流量 q (veh/h)', fontsize=12)
        ax3.set_title('流量-速度关系 (q-v图)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ 带多种拟合曲线的基本图已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_headway_distributions(self, headway_results, space_headway_results, output_dir=None):
        """
        绘制分车道的车头时距和车头间距分布图
        
        Args:
            headway_results: 车头时距结果字典 {lane: array}
            space_headway_results: 车头间距结果字典 {lane: array}
            output_dir: 输出目录
        """
        print("\n📊 开始绘制车头时距和车头间距分布图...")
        
        lanes = sorted([l for l in headway_results.keys() if len(headway_results[l]) > 0])
        
        if len(lanes) == 0:
            print("❌ 没有有效的车头时距数据")
            return
        
        # 设置颜色
        colors = plt.cm.Set1(np.linspace(0, 1, len(lanes)))
        lane_colors = dict(zip(lanes, colors))
        
        # 创建图形：2行，每行2列
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 车头时距直方图
        ax1 = axes[0, 0]
        for lane in lanes:
            headways = headway_results[lane]
            if len(headways) > 0:
                ax1.hist(headways, bins=30, alpha=0.6, label=f'车道 {lane}', 
                        color=lane_colors[lane], density=True)
        ax1.set_xlabel('车头时距 (秒)', fontsize=12)
        ax1.set_ylabel('概率密度', fontsize=12)
        ax1.set_title('车头时距分布', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 车头时距箱线图
        ax2 = axes[0, 1]
        headway_data = [headway_results[lane] for lane in lanes if len(headway_results[lane]) > 0]
        lane_labels = [f'车道 {lane}' for lane in lanes if len(headway_results[lane]) > 0]
        if headway_data:
            bp = ax2.boxplot(headway_data, labels=lane_labels, patch_artist=True)
            for patch, lane in zip(bp['boxes'], [l for l in lanes if len(headway_results[l]) > 0]):
                patch.set_facecolor(lane_colors[lane])
                patch.set_alpha(0.7)
        ax2.set_ylabel('车头时距 (秒)', fontsize=12)
        ax2.set_title('车头时距箱线图', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 车头间距直方图
        ax3 = axes[1, 0]
        space_lanes = sorted([l for l in space_headway_results.keys() if len(space_headway_results[l]) > 0])
        for lane in space_lanes:
            space_headways = space_headway_results[lane]
            if len(space_headways) > 0:
                ax3.hist(space_headways, bins=30, alpha=0.6, label=f'车道 {lane}', 
                        color=lane_colors.get(lane, 'gray'), density=True)
        ax3.set_xlabel('车头间距 (米)', fontsize=12)
        ax3.set_ylabel('概率密度', fontsize=12)
        ax3.set_title('车头间距分布', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 车头间距箱线图
        ax4 = axes[1, 1]
        space_headway_data = [space_headway_results[lane] for lane in space_lanes if len(space_headway_results[lane]) > 0]
        space_lane_labels = [f'车道 {lane}' for lane in space_lanes if len(space_headway_results[lane]) > 0]
        if space_headway_data:
            bp = ax4.boxplot(space_headway_data, labels=space_lane_labels, patch_artist=True)
            for patch, lane in zip(bp['boxes'], [l for l in space_lanes if len(space_headway_results[l]) > 0]):
                patch.set_facecolor(lane_colors.get(lane, 'gray'))
                patch.set_alpha(0.7)
        ax4.set_ylabel('车头间距 (米)', fontsize=12)
        ax4.set_title('车头间距箱线图', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, 'headway_distributions.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ 分布图已保存: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """主函数：处理两个CSV文件并进行分车道分析"""
    
    # 输入文件路径
    csv_file1 = '/home/lumos/Documents/traffic_analysis/data/processed/trajectories_cleaned_video1.csv'
    csv_file2 = '/home/lumos/Documents/traffic_analysis/data/processed/trajectories_cleaned_video2.csv'
    
    # 输出目录
    output_dir = '/home/lumos/Documents/traffic_analysis/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("           分车道交通流分析程序")
    print("=" * 80)
    
    # 读取数据
    print(f"\n📂 读取数据文件...")
    print(f"   文件1: {csv_file1}")
    print(f"   文件2: {csv_file2}")
    
    dfs = []
    for i, csv_file in enumerate([csv_file1, csv_file2], 1):
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['video_id'] = i  # 标记来源视频
            dfs.append(df)
            print(f"   ✅ 文件{i}: {len(df)} 行数据, {df['track_id'].nunique()} 辆唯一车辆")
        else:
            print(f"   ⚠️ 文件{i}不存在: {csv_file}")
    
    if len(dfs) == 0:
        print("❌ 没有可用的数据文件")
        return
    
    # 合并数据
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 合并后数据: {len(combined_df)} 行, {combined_df['track_id'].nunique()} 辆唯一车辆")
    
    # 创建分析器
    analyzer = LaneBasedTrafficAnalyzer(combined_df)
    
    # 1. 计算分车道交通流参数
    flow_params_df = analyzer.calculate_lane_flow_parameters()
    if not flow_params_df.empty:
        flow_params_path = os.path.join(output_dir, 'lane_flow_parameters.csv')
        flow_params_df.to_csv(flow_params_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 流量参数已保存: {flow_params_path}")
    
    # 2. 计算分车道车头时距
    headway_results = analyzer.calculate_lane_headway_distribution()
    
    # 3. 计算分车道车头间距
    space_headway_results = analyzer.calculate_lane_space_headway_distribution()
    
    # 4. 绘制交通流基本图
    if not flow_params_df.empty:
        fundamental_diagram_path = os.path.join(output_dir, 'lane_fundamental_diagram.png')
        analyzer.plot_lane_fundamental_diagram(flow_params_df, fundamental_diagram_path)
    
    # 5. 绘制车头时距和车头间距分布图
    analyzer.plot_headway_distributions(headway_results, space_headway_results, output_dir)
    
    # 6. 输出统计摘要
    print("\n" + "=" * 80)
    print("           分析结果摘要")
    print("=" * 80)
    
    if not flow_params_df.empty:
        print("\n📊 分车道交通流参数统计:")
        for lane in sorted(flow_params_df['lane'].unique()):
            lane_data = flow_params_df[flow_params_df['lane'] == lane]
            print(f"\n   车道 {lane}:")
            print(f"     平均流量: {lane_data['flow'].mean():.1f} veh/h")
            print(f"     平均速度: {lane_data['time_mean_speed'].mean():.2f} m/s")
            print(f"     平均密度: {lane_data['density'].mean():.1f} veh/km")
    
    print("\n📊 车头时距统计:")
    for lane in sorted(headway_results.keys()):
        headways = headway_results[lane]
        if len(headways) > 0:
            print(f"\n   车道 {lane}:")
            print(f"     样本数: {len(headways)}")
            print(f"     平均时距: {np.mean(headways):.2f} 秒")
            print(f"     标准差: {np.std(headways):.2f} 秒")
            print(f"     最小值: {np.min(headways):.2f} 秒")
            print(f"     最大值: {np.max(headways):.2f} 秒")
    
    print("\n📊 车头间距统计:")
    for lane in sorted(space_headway_results.keys()):
        space_headways = space_headway_results[lane]
        if len(space_headways) > 0:
            print(f"\n   车道 {lane}:")
            print(f"     样本数: {len(space_headways)}")
            print(f"     平均间距: {np.mean(space_headways):.2f} 米")
            print(f"     标准差: {np.std(space_headways):.2f} 米")
            print(f"     最小值: {np.min(space_headways):.2f} 米")
            print(f"     最大值: {np.max(space_headways):.2f} 米")
    
    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()

