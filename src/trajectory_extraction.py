import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
from tqdm import tqdm
import torch

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRACKING_CONFIG, PATH_CONFIG, LANE_CONFIG

class VehicleTracker:
    def __init__(self, model_path="/home/lumos/Documents/yolo11x.pt", vehicle_classes=None, conf_threshold=None):
        """车辆轨迹跟踪器"""
        
        # 使用配置参数或默认值
        self.model_path = model_path or TRACKING_CONFIG['model_path']
        self.vehicle_classes = vehicle_classes or TRACKING_CONFIG['vehicle_classes']
        self.conf_threshold = conf_threshold or TRACKING_CONFIG['conf_threshold']
        self.lane_config = LANE_CONFIG
        self.lane_boundaries = self.lane_config.get('boundaries', [])
        self.lane_labels = self.lane_config.get('labels', [])
        self.lane_mode = self.lane_config.get('mode', 'relative')
        self.lane_fallback = self.lane_config.get('fallback_label', 'unknown')
        
        # 初始化模型
        self.model = YOLO(self.model_path)
        self.trajectories = {}
        
        # 检查GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  使用设备: {self.device}")
    
    def process_video(self, video_path, output_path=None, skip_frames=1, target_fps=None):
        """
        处理视频并提取轨迹 - 修复时间计算问题
        """
        
        # 确保输出目录存在
        if output_path:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        print(f"📹 视频信息: {frame_width}x{frame_height}, FPS: {original_fps:.1f}, 总帧数: {total_frames}, 时长: {duration/60:.1f}分钟")
        
        # 修复：确保FPS不为零
        if original_fps <= 0:
            print("⚠️ 视频FPS异常，使用默认值30")
            original_fps = 30.0
        
        # 计算处理帧数
        processed_frames = (total_frames + skip_frames) // (skip_frames + 1)
        print(f"⚡ 处理设置: 跳帧={skip_frames}, 预计处理 {processed_frames} 帧")
        
        # 初始化轨迹数据存储
        trajectory_data = []
        frame_count = 0
        processed_count = 0
        
        # 创建进度条
        pbar = tqdm(total=processed_frames, desc="处理视频帧")
        start_time = time.time()
        
        # 性能统计
        performance_stats = {
            'frames_processed': 0,
            'vehicles_detected': 0,
            'processing_times': []
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 跳帧处理
            if frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
                
            if processed_count >= processed_frames:
                break
            
            # 调整图像尺寸以提高处理速度
            target_width = 640
            if frame_width > target_width:
                scale_factor = target_width / frame_width
                new_width = target_width
                new_height = int(frame_height * scale_factor)
                frame_resized = cv2.resize(frame, (new_width, new_height))
                width_scale = frame_width / new_width
                height_scale = frame_height / new_height
            else:
                frame_resized = frame
                width_scale = 1.0
                height_scale = 1.0
            
            frame_processing_start = time.time()
            
            # 运行YOLO检测
            results = self.model.track(
                frame_resized, 
                persist=True,
                classes=self.vehicle_classes,
                conf=self.conf_threshold,
                iou=TRACKING_CONFIG['iou_threshold'],
                verbose=False,
                tracker="bytetrack.yaml"
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # 直接从YOLO结果中提取检测信息（不需要supervision库）
                boxes = results[0].boxes
                
                # 获取所有检测框的信息
                track_ids = boxes.id.cpu().numpy().astype(int)  # 跟踪ID
                bboxes = boxes.xyxy.cpu().numpy()  # 边界框坐标 [x1, y1, x2, y2]
                class_ids = boxes.cls.cpu().numpy().astype(int)  # 类别ID
                confidences = boxes.conf.cpu().numpy()  # 置信度
                
                # 遍历每个检测结果
                for i in range(len(track_ids)):
                    track_id = track_ids[i]
                    bbox = bboxes[i]  # [x1, y1, x2, y2]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # 计算车辆中心点和边界框坐标
                    if width_scale != 1.0 or height_scale != 1.0:
                        # 还原到原始尺寸的坐标
                        bbox_x1 = bbox[0] * width_scale
                        bbox_y1 = bbox[1] * height_scale
                        bbox_x2 = bbox[2] * width_scale
                        bbox_y2 = bbox[3] * height_scale
                        center_x = ((bbox[0] + bbox[2]) / 2) * width_scale
                        center_y = ((bbox[1] + bbox[3]) / 2) * height_scale
                        bbox_width = (bbox[2] - bbox[0]) * width_scale
                        bbox_height = (bbox[3] - bbox[1]) * height_scale
                    else:
                        bbox_x1 = bbox[0]
                        bbox_y1 = bbox[1]
                        bbox_x2 = bbox[2]
                        bbox_y2 = bbox[3]
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        bbox_width = bbox[2] - bbox[0]
                        bbox_height = bbox[3] - bbox[1]
                    
                    # 修复：正确计算时间戳
                    # 使用 frame_count 而不是 processed_count 来计算时间
                    current_time = frame_count / original_fps
                    lane_label = self._assign_lane(center_x, frame_width)
                    
                    # 存储轨迹数据（包含完整信息）
                    trajectory_point = {
                        'frame_id': frame_count,
                        'time': current_time,  # 时间戳（秒）
                        'track_id': track_id,  # 车辆跟踪ID
                        'x': center_x,  # 中心点X坐标（像素）
                        'y': center_y,  # 中心点Y坐标（像素）
                        'bbox_x1': bbox_x1,  # 边界框左上角X坐标
                        'bbox_y1': bbox_y1,  # 边界框左上角Y坐标
                        'bbox_x2': bbox_x2,  # 边界框右下角X坐标
                        'bbox_y2': bbox_y2,  # 边界框右下角Y坐标
                        'width': bbox_width,  # 边界框宽度（像素）
                        'height': bbox_height,  # 边界框高度（像素）
                        'class_id': class_id,  # COCO类别ID
                        'vehicle_type': self._get_vehicle_type(class_id),  # 车辆类型（car/bus/truck/motorcycle）
                        'confidence': confidence,  # 检测置信度
                        'lane': lane_label  # 车道信息
                    }
                    trajectory_data.append(trajectory_point)
                    performance_stats['vehicles_detected'] += 1
            
            processed_count += 1
            frame_count += 1
            
            # 更新进度条
            pbar.update(1)
            
            # 计算处理时间
            frame_processing_time = time.time() - frame_processing_start
            performance_stats['processing_times'].append(frame_processing_time)
            
            # 每处理50帧更新一次性能信息
            if processed_count % 50 == 0:
                elapsed_time = time.time() - start_time
                current_fps = processed_count / elapsed_time
                avg_processing_time = np.mean(performance_stats['processing_times'][-50:])
                current_progress_minutes = frame_count / original_fps / 60
                
                pbar.set_postfix({
                    'fps': f'{current_fps:.1f}',
                    'vehicles': len(trajectory_data),
                    'frame_time': f'{avg_processing_time:.2f}s',
                    'progress': f'{current_progress_minutes:.1f}min'
                })
        
        pbar.close()
        cap.release()
        
        # 性能统计总结
        total_time = time.time() - start_time
        actual_fps = processed_count / total_time if total_time > 0 else 0
        final_time_minutes = frame_count / original_fps / 60
        
        print(f"✅ 处理完成!")
        print(f"📊 性能统计:")
        print(f"   - 总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
        print(f"   - 处理帧数: {processed_count}")
        print(f"   - 实际帧率: {actual_fps:.1f} FPS")
        print(f"   - 检测到轨迹点: {len(trajectory_data)}")
        print(f"   - 实际处理时长: {final_time_minutes:.1f} 分钟")
        if performance_stats['processing_times']:
            print(f"   - 平均每帧处理时间: {np.mean(performance_stats['processing_times']):.3f}秒")
        
        # 转换为DataFrame
        if len(trajectory_data) == 0:
            print("⚠️ 警告: 没有检测到任何车辆轨迹数据")
            return pd.DataFrame()
        
        trajectories_df = pd.DataFrame(trajectory_data)
        
        # 数据验证和统计
        unique_vehicles = trajectories_df['track_id'].nunique()
        total_points = len(trajectories_df)
        time_range = trajectories_df['time'].max() - trajectories_df['time'].min()
        
        print(f"\n📊 轨迹数据统计:")
        print(f"   - 总轨迹点数: {total_points}")
        print(f"   - 唯一车辆数: {unique_vehicles}")
        print(f"   - 时间范围: {trajectories_df['time'].min():.2f} - {trajectories_df['time'].max():.2f} 秒 ({time_range:.2f}秒)")
        print(f"   - 平均每车轨迹点: {total_points / unique_vehicles:.1f}" if unique_vehicles > 0 else "   - 平均每车轨迹点: 0")
        
        # 车辆类型统计
        if 'vehicle_type' in trajectories_df.columns:
            vehicle_type_counts = trajectories_df['vehicle_type'].value_counts()
            print(f"   - 车辆类型分布:")
            for vtype, count in vehicle_type_counts.items():
                print(f"     {vtype}: {count} 个轨迹点")
        
        # 保存轨迹数据到CSV文件
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 按列顺序保存（确保列的顺序一致）
            column_order = [
                'frame_id', 'time', 'track_id',
                'x', 'y',
                'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                'width', 'height',
                'class_id', 'vehicle_type', 'lane', 'confidence'
            ]
            
            # 只保存存在的列
            existing_columns = [col for col in column_order if col in trajectories_df.columns]
            trajectories_df[existing_columns].to_csv(
                output_path, 
                index=False,
                encoding='utf-8-sig'  # 使用UTF-8 BOM编码，确保Excel可以正确打开中文
            )
            
            print(f"\n💾 轨迹数据已保存至: {output_path}")
            print(f"   - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
            print(f"   - 数据列数: {len(existing_columns)}")
            print(f"   - 数据行数: {len(trajectories_df)}")
            
            # 保存数据摘要信息（可选）
            summary_path = output_path.replace('.csv', '_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("轨迹数据摘要信息\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"视频信息:\n")
                f.write(f"  - 分辨率: {frame_width}x{frame_height}\n")
                f.write(f"  - 帧率: {original_fps:.2f} FPS\n")
                f.write(f"  - 总帧数: {total_frames}\n")
                f.write(f"  - 时长: {duration/60:.2f} 分钟\n\n")
                f.write(f"处理设置:\n")
                f.write(f"  - 跳帧数: {skip_frames}\n")
                f.write(f"  - 处理帧数: {processed_count}\n")
                f.write(f"  - 置信度阈值: {self.conf_threshold}\n\n")
                f.write(f"轨迹数据统计:\n")
                f.write(f"  - 总轨迹点数: {total_points}\n")
                f.write(f"  - 唯一车辆数: {unique_vehicles}\n")
                f.write(f"  - 时间范围: {trajectories_df['time'].min():.2f} - {trajectories_df['time'].max():.2f} 秒\n")
                f.write(f"  - 时间跨度: {time_range:.2f} 秒 ({time_range/60:.2f} 分钟)\n")
                if unique_vehicles > 0:
                    f.write(f"  - 平均每车轨迹点: {total_points / unique_vehicles:.1f}\n")
                f.write(f"\n车辆类型分布:\n")
                if 'vehicle_type' in trajectories_df.columns:
                    for vtype, count in vehicle_type_counts.items():
                        f.write(f"  - {vtype}: {count} 个轨迹点 ({count/total_points*100:.1f}%)\n")
                f.write(f"\n性能统计:\n")
                f.write(f"  - 总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)\n")
                f.write(f"  - 实际帧率: {actual_fps:.1f} FPS\n")
                if performance_stats['processing_times']:
                    f.write(f"  - 平均每帧处理时间: {np.mean(performance_stats['processing_times']):.3f} 秒\n")
            
            print(f"📄 数据摘要已保存至: {summary_path}")
        
        return trajectories_df
    
    def process_video_fast(self, video_path, output_path=None):
        """快速处理模式 - 针对长视频优化"""
        # 创建临时tracker实例，使用更高的置信度阈值
        fast_tracker = VehicleTracker(
            model_path=self.model_path,
            vehicle_classes=self.vehicle_classes,
            conf_threshold=0.2  # 使用更高的置信度阈值
        )
        
        return fast_tracker.process_video(
            video_path=video_path,
            output_path=output_path,
            skip_frames=0  # 更高的跳帧
        )
    
    def process_video_balanced(self, video_path, output_path=None):
        """平衡处理模式"""
        # 创建临时tracker实例，使用适中的参数
        balanced_tracker = VehicleTracker(
            model_path=self.model_path,
            vehicle_classes=self.vehicle_classes,
            conf_threshold=0.1  # 适中的置信度阈值
        )
        
        return balanced_tracker.process_video(
            video_path=video_path,
            output_path=output_path,
            skip_frames=1  # 适中的跳帧
        )
    
    def _get_vehicle_type(self, class_id):
        """将类别ID映射为车辆类型"""
        vehicle_map = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return vehicle_map.get(class_id, 'other')
    
    def _assign_lane(self, center_x, frame_width):
        """根据中心点X坐标判断所在车道"""
        if not self.lane_boundaries or not self.lane_labels:
            return self.lane_fallback
        
        if len(self.lane_boundaries) != len(self.lane_labels) + 1:
            return self.lane_fallback
        
        if self.lane_mode == 'relative':
            if frame_width <= 0:
                return self.lane_fallback
            position = center_x / frame_width
        else:
            position = center_x
        
        for idx, label in enumerate(self.lane_labels):
            start = self.lane_boundaries[idx]
            end = self.lane_boundaries[idx + 1]
            if start <= position < end:
                return label
        
        return self.lane_labels[-1] if self.lane_labels else self.lane_fallback
    
    def generate_annotated_video(self, video_path, trajectories_df, output_video_path=None, 
                                 show_track_id=True, show_vehicle_type=True, show_confidence=True):
        """
        生成标注视频，在视频上显示检测框、跟踪ID、车辆类型等信息
        
        Args:
            video_path: 输入视频路径
            trajectories_df: 轨迹数据DataFrame
            output_video_path: 输出视频路径（如果为None，自动生成）
            show_track_id: 是否显示跟踪ID
            show_vehicle_type: 是否显示车辆类型
            show_confidence: 是否显示置信度
        
        Returns:
            输出视频路径
        """
        
        if trajectories_df.empty:
            print("⚠️ 轨迹数据为空，无法生成标注视频")
            return None
        
        # 如果没有指定输出路径，自动生成
        if output_video_path is None:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(video_dir, f"{video_name}_annotated.mp4")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🎬 开始生成标注视频...")
        print(f"   输入视频: {video_path}")
        print(f"   输出视频: {output_video_path}")
        
        # 打开输入视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        # 尝试使用H.264编码（更好的压缩率和兼容性），如果失败则使用mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"⚠️ H.264编码不可用，使用mp4v编码...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print(f"❌ 无法创建视频文件: {output_video_path}")
                cap.release()
                return None
        
        # 车辆类型颜色映射
        vehicle_colors = {
            'car': (0, 255, 0),        # 绿色
            'bus': (255, 0, 0),         # 蓝色
            'truck': (0, 0, 255),      # 红色
            'motorcycle': (255, 255, 0), # 青色
            'other': (128, 128, 128)    # 灰色
        }
        
        # 按帧分组轨迹数据
        trajectories_by_frame = {}
        for _, row in trajectories_df.iterrows():
            frame_id = int(row['frame_id'])
            if frame_id not in trajectories_by_frame:
                trajectories_by_frame[frame_id] = []
            trajectories_by_frame[frame_id].append(row)
        
        # 创建进度条
        pbar = tqdm(total=total_frames, desc="生成标注视频")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在当前帧上绘制标注
            if frame_count in trajectories_by_frame:
                for _, vehicle in enumerate(trajectories_by_frame[frame_count]):
                    # 获取车辆信息
                    track_id = int(vehicle['track_id'])
                    bbox_x1 = int(vehicle['bbox_x1'])
                    bbox_y1 = int(vehicle['bbox_y1'])
                    bbox_x2 = int(vehicle['bbox_x2'])
                    bbox_y2 = int(vehicle['bbox_y2'])
                    vehicle_type = vehicle['vehicle_type']
                    confidence = vehicle['confidence']
                    
                    # 获取车辆颜色
                    color = vehicle_colors.get(vehicle_type, (128, 128, 128))
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, 2)
                    
                    # 准备标签文本
                    label_parts = []
                    if show_track_id:
                        label_parts.append(f"ID:{track_id}")
                    if show_vehicle_type:
                        label_parts.append(vehicle_type)
                    if show_confidence:
                        label_parts.append(f"{confidence:.2f}")
                    
                    label = " ".join(label_parts)
                    
                    # 计算文本位置（在边界框上方）
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    text_x = bbox_x1
                    text_y = max(bbox_y1 - 5, text_height)
                    
                    # 绘制文本背景（半透明）
                    cv2.rectangle(
                        frame,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width, text_y + baseline),
                        color,
                        -1
                    )
                    
                    # 绘制文本
                    cv2.putText(
                        frame,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),  # 白色文字
                        1,
                        cv2.LINE_AA
                    )
            
            # 在视频左上角添加帧信息
            info_text = f"Frame: {frame_count} | Time: {frame_count/fps:.2f}s"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # 写入帧
            out.write(frame)
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        # 检查输出文件
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path) / 1024 / 1024
            print(f"\n✅ 标注视频生成完成!")
            print(f"   - 输出文件: {output_video_path}")
            print(f"   - 文件大小: {file_size:.2f} MB")
            print(f"   - 分辨率: {width}x{height}")
            print(f"   - 帧率: {fps:.2f} FPS")
            print(f"   - 总帧数: {frame_count}")
            return output_video_path
        else:
            print(f"❌ 标注视频生成失败")
            return None


def main():
    """主函数：提取车辆轨迹并保存到CSV文件"""
    
    # 输入视频路径
    video_path = '/home/lumos/Documents/traffic_analysis/data/raw_videos/traffic_video.mp4'
    
    # 输出CSV文件路径（相对于当前文件位置）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '../data/processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trajectories_raw.csv')
    
    # 转换为绝对路径
    output_path = os.path.abspath(output_path)
    
    print("=" * 60)
    print("          车辆轨迹提取程序")
    print("=" * 60)
    print(f"\n📹 输入视频: {video_path}")
    print(f"💾 输出文件: {output_path}")
    print(f"📁 输出目录: {os.path.dirname(output_path)}")
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"\n❌ 错误: 视频文件不存在!")
        print(f"   路径: {video_path}")
        return
    
    # 创建跟踪器
    print(f"\n🔧 初始化车辆跟踪器...")
    tracker = VehicleTracker()
    
    # 根据视频长度选择处理模式
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"📊 视频信息: 时长 {duration/60:.1f} 分钟")
    
    # 选择处理模式
    if duration > 10 * 60:  # 超过10分钟
        print("⚡ 使用快速模式处理长视频...")
        trajectories_df = tracker.process_video_fast(video_path, output_path)
    elif duration > 2 * 60:  # 2-10分钟
        print("⚖️ 使用平衡模式...")
        trajectories_df = tracker.process_video_balanced(video_path, output_path)
    else:  # 短视频
        print("📝 使用标准模式...")
        trajectories_df = tracker.process_video(video_path, output_path)
    
    # 显示结果
    if trajectories_df.empty:
        print("\n❌ 没有提取到轨迹数据，请检查:")
        print("   1. 视频中是否有车辆")
        print("   2. 置信度阈值是否设置过高")
        print("   3. 车辆类别设置是否正确")
    else:
        print("\n" + "=" * 60)
        print("✅ 轨迹提取完成!")
        print("=" * 60)
        print(f"\n📊 结果统计:")
        print(f"   - 输出文件: {output_path}")
        print(f"   - 文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"   - 总轨迹点数: {len(trajectories_df)}")
        print(f"   - 唯一车辆数: {trajectories_df['track_id'].nunique()}")
        print(f"   - 时间范围: {trajectories_df['time'].min():.2f} - {trajectories_df['time'].max():.2f} 秒")
        print(f"   - 时间跨度: {(trajectories_df['time'].max() - trajectories_df['time'].min())/60:.2f} 分钟")
        
        # 车辆类型统计
        if 'vehicle_type' in trajectories_df.columns:
            vehicle_type_counts = trajectories_df['vehicle_type'].value_counts()
            print(f"\n🚗 车辆类型分布:")
            for vtype, count in vehicle_type_counts.items():
                percentage = count / len(trajectories_df) * 100
                print(f"   - {vtype}: {count} 个轨迹点 ({percentage:.1f}%)")
        
        # 摘要文件路径
        summary_path = output_path.replace('.csv', '_summary.txt')
        if os.path.exists(summary_path):
            print(f"\n📄 数据摘要: {summary_path}")
        
        # 生成标注视频
        print(f"\n🎬 是否生成标注视频? (需要重新处理视频，可能需要较长时间)")
        print(f"   正在生成标注视频...")
        
        # 设置标注视频输出路径
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        annotated_video_path = os.path.join(
            os.path.dirname(output_path), 
            f"{video_name}_annotated.mp4"
        )
        annotated_video_path = os.path.abspath(annotated_video_path)
        
        # 生成标注视频
        annotated_video = tracker.generate_annotated_video(
            video_path=video_path,
            trajectories_df=trajectories_df,
            output_video_path=annotated_video_path,
            show_track_id=True,
            show_vehicle_type=True,
            show_confidence=True
        )
        
        if annotated_video:
            print(f"\n📹 标注视频已保存: {annotated_video}")
        
        print(f"\n💡 提示: 可以使用以下代码加载数据:")
        print(f"   import pandas as pd")
        print(f"   df = pd.read_csv('{output_path}')")


if __name__ == "__main__":
    main()
