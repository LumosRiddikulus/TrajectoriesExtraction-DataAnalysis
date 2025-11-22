"""
轨迹数据清洗脚本
================
针对 `trajectories_raw_video1/2.csv` 中存在的异常纵向坐标、短碎轨迹、
噪声较大的定位等问题，执行以下步骤：
1. 补齐世界坐标 (x_world, y_world)
2. 过滤超出道路范围和含 NaN/Inf 的点
3. 删除轨迹点数过少或瞬间跳变过大的轨迹
4. 固定时间步长重采样 + 线性插值
5. 滚动平均平滑
6. 一维常速卡尔曼滤波
清洗后的数据将分别保存为 `trajectories_cleaned_videoX.csv`。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from coordinate_transformation import CoordinateTransformer
from config import ROAD_CONFIG


LANE_MAPPING = {
    "lan1": 1,
    "lane1": 1,
    "1": 1,
    "lan2": 2,
    "lane2": 2,
    "2": 2,
    "lan3": 3,
    "lane3": 3,
    "3": 3,
    "non-motor": 0,
    "non_motor": 0,
    "nonmotor": 0,
    "0": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗轨迹数据")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/home/lumos/Documents/traffic_analysis/data/processed/trajectories_raw_video1.csv",
            "/home/lumos/Documents/traffic_analysis/data/processed/trajectories_raw_video2.csv",
        ],
        help="输入 CSV 路径，默认包含 video1 和 video2。",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lumos/Documents/traffic_analysis/data/processed",
        help="输出目录，默认与原数据相同。",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.2,
        help="重采样时间步长 (秒)。",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=15,
        help="轨迹最小点数，小于该值将被丢弃。",
    )
    parser.add_argument(
        "--max-jump",
        type=float,
        default=5.0,
        help="单步允许的最大位移 (米)，超过视为异常点。",
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["source_file"] = path.name
    print(f"✅ 读取 {path.name}: {len(df)} 行, {df['track_id'].nunique()} 条轨迹")
    return df


def ensure_world_coords(df: pd.DataFrame, transformer: CoordinateTransformer) -> pd.DataFrame:
    if {"x_world", "y_world"}.issubset(df.columns):
        return df
    print("🔄 未找到世界坐标，执行透视变换...")
    world = transformer.pixel_to_world(df[["x", "y"]].values)
    df["x_world"] = world[:, 0]
    df["y_world"] = world[:, 1]
    return df


def clean_range(df: pd.DataFrame, margin: float = 5.0) -> pd.DataFrame:
    lower = -margin
    upper = ROAD_CONFIG.get("length", 0) + margin
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x_world", "time"])
    df = df[(df["x_world"] >= lower) & (df["x_world"] <= upper)]
    removed = before - len(df)
    if removed:
        print(f"🧹 去除越界/缺失点 {removed} 行")
    return df


def standardize_lane(df: pd.DataFrame) -> pd.DataFrame:
    if "lane" not in df.columns:
        df["lane_num"] = 0
    else:
        df["lane_num"] = (
            df["lane"].astype(str).str.lower().map(LANE_MAPPING).fillna(0).astype(int)
        )
    return df


def basic_filter(df: pd.DataFrame, min_points: int, max_jump: float) -> pd.DataFrame:
    groups: List[pd.DataFrame] = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time")
        if len(g) < min_points:
            continue
        jumps = g["x_world"].diff().abs()
        g = g[(jumps <= max_jump) | jumps.isna()]
        if len(g) < min_points:
            continue
        groups.append(g)
    if not groups:
        return pd.DataFrame(columns=df.columns)
    result = pd.concat(groups, ignore_index=True)
    print(f"🧽 基础过滤后 {len(result)} 行, {result['track_id'].nunique()} 条轨迹")
    return result


def resample_tracks(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    resampled = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time")
        if len(g) < 2:
            continue
        start, end = g["time"].iloc[0], g["time"].iloc[-1]
        if end - start < dt:
            continue
        new_times = np.arange(start, end + 1e-9, dt)
        new_x = np.interp(new_times, g["time"], g["x_world"])
        new_y = (
            np.interp(new_times, g["time"], g["y_world"])
            if "y_world" in g.columns
            else np.zeros_like(new_times)
        )
        lane_num = g["lane_num"].iloc[0] if "lane_num" in g else 0
        lane_label = g["lane"].iloc[0] if "lane" in g else str(lane_num)
        vehicle_type = g["vehicle_type"].iloc[0] if "vehicle_type" in g else "car"
        src = g["source_file"].iloc[0]

        resampled.append(
            pd.DataFrame(
                {
                    "track_id": tid,
                    "time": new_times,
                    "x_world": new_x,
                    "y_world": new_y,
                    "lane": lane_label,
                    "lane_num": lane_num,
                    "vehicle_type": vehicle_type,
                    "source_file": src,
                }
            )
        )
    if not resampled:
        return pd.DataFrame(columns=["track_id", "time", "x_world", "y_world"])
    result = pd.concat(resampled, ignore_index=True)
    print(f"🔁 重采样 (dt={dt}s) 后 {len(result)} 行, {result['track_id'].nunique()} 条轨迹")
    return result


def smooth_tracks(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    smoothed = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time")
        g["x_world"] = g["x_world"].rolling(window, center=True, min_periods=1).mean()
        smoothed.append(g)
    return pd.concat(smoothed, ignore_index=True)


def kalman_filter(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    filtered = []
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    q = 0.5
    r = 1.0
    Q = q * np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]])
    R = np.array([[r]])

    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time").copy()
        z = g["x_world"].values
        x_state = np.array([z[0], 0.0])
        P = np.eye(2)
        outputs = []
        for measurement in z:
            x_state = F @ x_state
            P = F @ P @ F.T + Q

            y = measurement - (H @ x_state)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x_state = x_state + (K @ y).flatten()
            P = (np.eye(2) - K @ H) @ P

            outputs.append(x_state[0])

        g["x_world"] = outputs
        filtered.append(g)

    result = pd.concat(filtered, ignore_index=True)
    print("🤖 卡尔曼滤波完成")
    return result


def process_file(path: Path, args: argparse.Namespace, transformer: CoordinateTransformer) -> Path:
    df = load_csv(path)
    df = ensure_world_coords(df, transformer)
    df = clean_range(df)
    df = standardize_lane(df)
    df = basic_filter(df, args.min_points, args.max_jump)
    df = resample_tracks(df, args.dt)
    df = smooth_tracks(df)
    df = kalman_filter(df, args.dt)

    output_path = Path(args.output_dir) / path.name.replace("raw", "cleaned")
    df.to_csv(output_path, index=False)
    print(f"💾 清洗结果已保存: {output_path} ({len(df)} 行)")
    return output_path


def main():
    args = parse_args()
    transformer = CoordinateTransformer()
    output_files = []
    for input_path in args.inputs:
        output_files.append(process_file(Path(input_path), args, transformer))

    print("\n=== 清洗完成 ===")
    for file in output_files:
        print(f" - {file}")


if __name__ == "__main__":
    main()
