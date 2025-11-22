"""
生成时空轨迹图
-----------------
读取两个原始轨迹CSV文件，必要时利用透视变换补齐 x_world / y_world，
并将每辆车的纵向位置随时间的轨迹绘制在同一张时空图中。

用法示例：
    python src/plot_spacetime_trajectories.py \
        --input data/processed/trajectories_raw_video1.csv \
                data/processed/trajectories_raw_video2.csv \
        --output data/processed/spacetime_trajectories.png \
        --lanes 1 2 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

from config import ROAD_CONFIG
from coordinate_transformation import CoordinateTransformer

# 输入文件默认指向用户给定的CSV
DEFAULT_INPUT_FILES = [
    "/home/lumos/Documents/traffic_analysis/data/processed/trajectories_raw_video1.csv",
    "/home/lumos/Documents/traffic_analysis/data/processed/trajectories_raw_video2.csv",
]

DEFAULT_OUTPUT = (
    "/home/lumos/Documents/traffic_analysis/data/processed/spacetime_trajectories.png"
)

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

LANE_COLORS = {
    1: "#d62728",  # red
    2: "#ff7f0e",  # orange
    3: "#7f7f7f",  # gray
    0: "#2ca02c",  # green, e.g. non-motor
}


def setup_chinese_font() -> None:
    """设置中文字体，优先使用系统可用的字体，避免图中出现方块字。"""
    chinese_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans CN",
        "STHeiti",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    if not selected_font:
        cjk_fonts = [
            f.name
            for f in fm.fontManager.ttflist
            if any(tag in f.name for tag in ("CJK", "Chinese", "SC", "CN"))
        ]
        if cjk_fonts:
            selected_font = cjk_fonts[0]

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font] + plt.rcParams["font.sans-serif"]
        print(f"✅ 已设置中文字体: {selected_font}")
    else:
        print("⚠️ 未找到中文字体，可能会出现方块字。")

    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制时空轨迹图")
    parser.add_argument(
        "--input",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
        help="输入轨迹CSV文件路径，默认使用提供的两个原始文件。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="输出图片路径，默认为 data/processed/spacetime_trajectories.png。",
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        type=int,
        help="仅绘制指定车道（例如 --lanes 1 2）。不设置则绘制所有车道。",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="限定时间范围（秒），例如 --time-range 0 300。",
    )
    parser.add_argument(
        "--space-range",
        nargs=2,
        type=float,
        metavar=("MIN_X", "MAX_X"),
        help="限定空间范围（米），例如 --space-range 0 50。",
    )
    parser.add_argument(
        "--max-trajectories-per-lane",
        type=int,
        default=None,
        help="每个车道最多绘制多少条轨迹，用于避免图像过密。",
    )
    return parser.parse_args()


def load_trajectories(csv_paths: Sequence[str]) -> pd.DataFrame:
    """读取并合并多个CSV文件。"""
    dataframes: List[pd.DataFrame] = []

    for path in csv_paths:
        if not path:
            continue
        csv_path = Path(path)
        if not csv_path.exists():
            print(f"⚠️  未找到文件: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["source_file"] = csv_path.name
        dataframes.append(df)
        print(f"✅ 读取 {csv_path.name}: {len(df)} 行, {df['track_id'].nunique()} 辆车")

    if not dataframes:
        raise FileNotFoundError("未能读取到任何轨迹数据，请检查输入路径。")

    combined = pd.concat(dataframes, ignore_index=True)
    print(
        f"📊 合并后共 {len(combined)} 行, "
        f"{combined['track_id'].nunique()} 个 track_id"
    )
    return combined


def ensure_world_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """如果不存在世界坐标，则根据像素坐标进行转换。"""
    if {"x_world", "y_world"}.issubset(df.columns):
        return df

    print("🔄 未找到 x_world / y_world，执行坐标转换...")
    transformer = CoordinateTransformer()
    world_coords = transformer.pixel_to_world(df[["x", "y"]].values)
    df["x_world"] = world_coords[:, 0]
    df["y_world"] = world_coords[:, 1]
    return df


def standardize_lane_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """将lane列映射为数值 lane_num，便于筛选与着色。"""
    if "lane" not in df.columns:
        df["lane_num"] = 0
        return df

    df["lane_num"] = (
        df["lane"]
        .astype(str)
        .str.lower()
        .map(LANE_MAPPING)
        .fillna(0)
        .astype(int)
    )
    return df


def clean_world_coordinates(df: pd.DataFrame, margin: float = 5.0) -> pd.DataFrame:
    """
    去除明显超出道路范围的横向坐标，避免异常轨迹。

    Args:
        margin: 在道路长度基础上的上下限缓冲（米）
    """
    if df.empty or "x_world" not in df.columns:
        return df

    road_length = ROAD_CONFIG.get("length", 0)
    lower = -margin
    upper = road_length + margin if road_length > 0 else df["x_world"].quantile(0.99)

    cleaned = df.replace([np.inf, -np.inf], np.nan)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["x_world", "time"])
    cleaned = cleaned[(cleaned["x_world"] >= lower) & (cleaned["x_world"] <= upper)]
    removed = before - len(cleaned)
    if removed > 0:
        print(f"🧹 清理异常坐标 {removed} 行 (范围 {lower:.1f}~{upper:.1f} m)")
    return cleaned


def basic_cleaning(df: pd.DataFrame, min_points: int = 15, max_jump: float = 5.0) -> pd.DataFrame:
    """按轨迹清洗：去掉过短轨迹与瞬时跳变。"""
    if df.empty:
        return df

    cleaned_groups = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("time")
        if len(group) < min_points:
            continue
        jumps = group["x_world"].diff().abs()
        group = group[(jumps <= max_jump) | jumps.isna()]
        if len(group) < min_points:
            continue
        cleaned_groups.append(group)

    if not cleaned_groups:
        return pd.DataFrame(columns=df.columns)

    cleaned_df = pd.concat(cleaned_groups, ignore_index=True)
    print(f"🧽 基础清洗后剩余 {len(cleaned_df)} 行, {cleaned_df['track_id'].nunique()} 条轨迹")
    return cleaned_df


def resample_and_interpolate(
    df: pd.DataFrame, dt: float = 0.2
) -> pd.DataFrame:
    """对每条轨迹按固定步长重采样并线性插值。"""
    if df.empty:
        return df

    resampled_groups = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("time")
        if len(group) < 2:
            continue
        start, end = group["time"].iloc[0], group["time"].iloc[-1]
        if end - start < dt:
            continue
        new_times = np.arange(start, end + 1e-9, dt)
        new_x = np.interp(new_times, group["time"], group["x_world"])
        lane = group["lane_num"].iloc[0]
        source = group["source_file"].iloc[0]
        resampled_groups.append(
            pd.DataFrame(
                {
                    "track_id": track_id,
                    "time": new_times,
                    "x_world": new_x,
                    "lane_num": lane,
                    "source_file": source,
                }
            )
        )

    if not resampled_groups:
        return pd.DataFrame(columns=df.columns)

    resampled_df = pd.concat(resampled_groups, ignore_index=True)
    print(
        f"🔁 重采样 (dt={dt}s) 后 {len(resampled_df)} 行, "
        f"{resampled_df['track_id'].nunique()} 条轨迹"
    )
    return resampled_df


def smooth_tracks(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """对轨迹应用滚动平均平滑。"""
    if df.empty:
        return df

    smoothed_groups = []
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("time")
        group["x_world"] = (
            group["x_world"].rolling(window, center=True, min_periods=1).mean()
        )
        smoothed_groups.append(group)

    smoothed_df = pd.concat(smoothed_groups, ignore_index=True)
    print(f"🌊 滚动平滑 (window={window}) 完成")
    return smoothed_df


def apply_kalman_filter(
    df: pd.DataFrame,
    dt: float = 0.2,
    process_var: float = 0.5,
    measurement_var: float = 1.0,
) -> pd.DataFrame:
    """对每条轨迹应用一维常速卡尔曼滤波，输出平滑位置。"""
    if df.empty:
        return df

    filtered_groups = []
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = process_var * np.array(
        [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]
    )
    R = np.array([[measurement_var]])

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("time").copy()
        z = group["x_world"].values
        x_state = np.array([z[0], 0.0])
        P = np.eye(2)
        filtered_positions = []

        for measurement in z:
            # predict
            x_state = F @ x_state
            P = F @ P @ F.T + Q

            # update
            y = measurement - (H @ x_state)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x_state = x_state + (K @ y).flatten()
            P = (np.eye(2) - K @ H) @ P
            filtered_positions.append(x_state[0])

        group["x_world"] = filtered_positions
        filtered_groups.append(group)

    filtered_df = pd.concat(filtered_groups, ignore_index=True)
    print("🤖 卡尔曼滤波完成")
    return filtered_df


def filter_dataframe(
    df: pd.DataFrame,
    lanes: Optional[Iterable[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    space_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """按车道、时间、空间范围筛选数据。"""
    filtered = df.copy()

    if lanes:
        lane_set = set(lanes)
        filtered = filtered[filtered["lane_num"].isin(lane_set)]
        print(f"🔎 过滤车道 {sorted(lane_set)} 后剩余 {len(filtered)} 行")

    if time_range:
        t0, t1 = time_range
        filtered = filtered[(filtered["time"] >= t0) & (filtered["time"] <= t1)]
        print(f"⏱️  时间范围 {t0}–{t1}s -> {len(filtered)} 行")

    if space_range:
        x0, x1 = space_range
        filtered = filtered[(filtered["x_world"] >= x0) & (filtered["x_world"] <= x1)]
        print(f"📏 空间范围 {x0}–{x1}m -> {len(filtered)} 行")

    return filtered


def plot_spacetime_trajectories(
    df: pd.DataFrame,
    output_path: str,
    max_traj_per_lane: Optional[int] = None,
) -> None:
    """绘制时空图并保存。"""
    if df.empty:
        raise ValueError("没有可用的数据，无法绘制时空轨迹图。")

    plt.figure(figsize=(14, 8))

    for lane, lane_df in sorted(df.groupby("lane_num")):
        # 按车道拆分后，再按track绘制
        track_groups = list(lane_df.groupby("track_id"))
        if max_traj_per_lane is not None and len(track_groups) > max_traj_per_lane:
            track_groups = track_groups[:max_traj_per_lane]
            print(
                f"  ✂️  车道 {lane} 轨迹过多，仅绘制前 {max_traj_per_lane} 条 "
                f"(按 track_id 升序)"
            )

        color = LANE_COLORS.get(lane, "#808080")
        for track_id, vehicle in track_groups:
            vehicle = vehicle.sort_values("time")
            plt.plot(
                vehicle["time"].values,
                vehicle["x_world"].values,
                color=color,
                alpha=0.45,
                linewidth=1.2,
            )

    plt.xlabel("时间 t (秒)")
    plt.ylabel("沿路段位置 x (米)")
    plt.title("时空轨迹图")
    plt.grid(True, alpha=0.3)

    # 提示车道颜色
    legend_entries = []
    for lane in sorted(df["lane_num"].unique()):
        legend_entries.append(
            plt.Line2D(
                [0],
                [0],
                color=LANE_COLORS.get(lane, "#808080"),
                lw=3,
                label=f"车道 {lane}",
            )
        )
    if legend_entries:
        plt.legend(handles=legend_entries, title="车道", loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ 时空轨迹图已保存: {output_path}")


def plot_spacetime_per_lane(
    df: pd.DataFrame,
    output_dir: str,
    max_traj_per_lane: Optional[int] = None,
) -> None:
    """为每个车道单独绘制时空图。"""
    os.makedirs(output_dir, exist_ok=True)
    for lane in sorted(df["lane_num"].unique()):
        lane_df = df[df["lane_num"] == lane]
        if lane_df.empty:
            continue

        track_groups = list(lane_df.groupby("track_id"))
        if max_traj_per_lane is not None and len(track_groups) > max_traj_per_lane:
            track_groups = track_groups[:max_traj_per_lane]
            print(f"  ✂️ 车道 {lane} 限制为 {max_traj_per_lane} 条轨迹")

        plt.figure(figsize=(12, 6))
        color = LANE_COLORS.get(lane, "#808080")
        for _, vehicle in track_groups:
            vehicle = vehicle.sort_values("time")
            if len(vehicle) < 2:
                continue
            plt.plot(
                vehicle["time"].values,
                vehicle["x_world"].values,
                color=color,
                alpha=0.6,
                linewidth=1.4,
            )

        plt.xlabel("时间 t (秒)")
        plt.ylabel("沿路段位置 x (米)")
        plt.title(f"车道 {lane} 时空轨迹图")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        lane_path = Path(output_dir) / f"spacetime_trajectories_lane{lane}.png"
        plt.savefig(lane_path, dpi=300)
        plt.close()
        print(f"✅ 车道 {lane} 图已保存: {lane_path}")


def main():
    args = parse_args()
    setup_chinese_font()

    df = load_trajectories(args.input)
    df = ensure_world_coordinates(df)
    df = standardize_lane_numbers(df)
    df = clean_world_coordinates(df)

    df = filter_dataframe(
        df,
        lanes=args.lanes,
        time_range=tuple(args.time_range) if args.time_range else None,
        space_range=tuple(args.space_range) if args.space_range else None,
    )
    df = basic_cleaning(df)
    df = resample_and_interpolate(df)
    df = smooth_tracks(df)
    df = apply_kalman_filter(df)

    plot_spacetime_trajectories(
        df,
        output_path=args.output,
        max_traj_per_lane=args.max_trajectories_per_lane,
    )
    output_dir = os.path.dirname(args.output) or "."
    plot_spacetime_per_lane(
        df,
        output_dir=output_dir,
        max_traj_per_lane=args.max_trajectories_per_lane,
    )


if __name__ == "__main__":
    main()
