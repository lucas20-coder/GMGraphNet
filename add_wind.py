#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从ERA5多层数据智能计算台风风速（增强版）- 节（kt）单位版本
- 解决高度差异（850hPa → 10m）
- 考虑台风径向结构
- 多层信息融合
- 质量控制
- 与真实值对比
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from scipy.interpolate import interp1d

# ==================== 配置 ====================

INPUT_CSV = r"f:/era5_output/typhoon_grid_50x50_0.25deg_1980Q1-2011Q4_multiprocess.csv"
OUTPUT_CSV = r"E:/era5_output/typhoon_grid_50x50_0.25deg_1965Q1-1986Q4_real_wind.csv"
IBTRACS_CSV = r"F:/ibtracs_1965Q1_1986Q4.csv"  # 真实值对比（可选）

CHUNK_SIZE = 600000

# ==================== 气象学参数 ====================

# 1. 高度订正系数（从850hPa到10m）
# 基于经验关系：V_10m = V_850hPa * correction_factor
HEIGHT_CORRECTION = 0.70  # 850hPa风速约为10m的1.43倍，反过来乘0.70

# 2. 台风最大风速半径范围
RMW_MIN = 15  # km
RMW_MAX = 80  # km

# 3. 质量控制参数（单位改为kt）
MAX_REASONABLE_WIND = 175  # kt (约90m/s，超强台风上限)
MIN_REASONABLE_WIND = 10  # kt (约5m/s，热带低压下限)
MAX_HOURLY_CHANGE = 30  # kt/h (最大小时变化率，15m/s/h约为30kt/h)

# 单位转换常数
MS_TO_KT = 1.94384  # 1 m/s = 1.94384 kt
KT_TO_MS = 0.514444  # 1 kt = 0.514444 m/s


def load_ibtracs_for_comparison(ibtracs_csv):
    """加载IBTrACS真实风速用于对比验证（可选）"""
    if not os.path.exists(ibtracs_csv):
        print(f"⚠️ IBTrACS文件不存在，跳过对比验证")
        return None

    try:
        print(f"\n加载 IBTrACS 数据用于验证...")
        ibtracs = pd.read_csv(ibtracs_csv)

        # 时间转换
        if 'ISO_TIME' in ibtracs.columns:
            ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
        else:
            ibtracs['time'] = pd.to_datetime(ibtracs['time'])

        # 移除时区
        if ibtracs['time'].dt.tz is not None:
            ibtracs['time'] = ibtracs['time'].dt.tz_localize(None)

        # 风速列（直接使用kt单位）
        if 'WMO_WIND' in ibtracs.columns:
            ibtracs['true_wind_kt'] = pd.to_numeric(ibtracs['WMO_WIND'], errors='coerce')
        elif 'USA_WIND' in ibtracs.columns:
            ibtracs['true_wind_kt'] = pd.to_numeric(ibtracs['USA_WIND'], errors='coerce')
        else:
            print(f"⚠️ 未找到风速列")
            return None

        # storm_id
        if 'SID' in ibtracs.columns:
            ibtracs['storm_id'] = ibtracs['SID']

        # 格式化时间（与ERA5对齐）
        ibtracs['time_str'] = ibtracs['time'].dt.strftime('%Y/%m/%d %H:%M')

        # 创建查找字典
        ibtracs_dict = {}
        for _, row in ibtracs.iterrows():
            key = (str(row['storm_id']), str(row['time_str']))
            ibtracs_dict[key] = row['true_wind_kt']

        print(f"✓ 加载 {len(ibtracs_dict):,} 个真实风速值（单位：kt）")
        return ibtracs_dict

    except Exception as e:
        print(f"⚠️ 加载 IBTrACS 失败: {e}")
        return None


def calculate_multi_level_wind(row, u_cols, v_cols):
    """
    多层风速计算（考虑垂直结构）
    返回单位：m/s（后续会转换为kt）
    """
    winds = []
    weights = []

    # 层次权重（850hPa最重要）
    level_weights = {
        '850hPa': 0.50,  # 台风主要层次
        '750hPa': 0.25,  # 中层
        '500hPa': 0.15,  # 中上层
        '200hPa': 0.10,  # 高层辐散
    }

    for level, weight in level_weights.items():
        u_col = f'U_GRD_L100_{level}'
        v_col = f'V_GRD_L100_{level}'

        if u_col in u_cols and v_col in v_cols:
            try:
                u = pd.to_numeric(row[u_col], errors='coerce')
                v = pd.to_numeric(row[v_col], errors='coerce')

                if not np.isnan(u) and not np.isnan(v):
                    wind = np.sqrt(u ** 2 + v ** 2)
                    winds.append(wind)
                    weights.append(weight)
            except:
                continue

    if len(winds) == 0:
        return np.nan

    # 加权平均
    weighted_wind = np.average(winds, weights=weights)

    return weighted_wind


def calculate_vorticity_based_wind(row):
    """
    基于涡度估算风速（辅助方法）
    返回单位：m/s（后续会转换为kt）
    """
    vort_850 = pd.to_numeric(row.get('VORT_L100_850hPa', np.nan), errors='coerce')
    abs_vort_850 = pd.to_numeric(row.get('ABS_V_L100_850hPa', np.nan), errors='coerce')

    if not np.isnan(vort_850) and not np.isnan(abs_vort_850):
        # 经验公式（需要根据实际情况调整）
        vort_wind = abs_vort_850 * 8000  # 涡度转风速
        return np.clip(vort_wind, 0, 100)

    return np.nan


def calculate_enhanced_wind(chunk):
    """
    增强版风速计算（单位改为kt）
    """

    # 可用的U/V列
    u_cols = [col for col in chunk.columns if 'U_GRD' in col]
    v_cols = [col for col in chunk.columns if 'V_GRD' in col]

    # 1. 多层风速计算（单位：m/s）
    chunk['wind_multi_level_ms'] = chunk.apply(
        lambda row: calculate_multi_level_wind(row, u_cols, v_cols),
        axis=1
    )

    # 转换为kt
    chunk['wind_multi_level'] = chunk['wind_multi_level_ms'] * MS_TO_KT

    # 2. 高度订正（850hPa → 10m）
    chunk['wind_10m'] = chunk['wind_multi_level'] * HEIGHT_CORRECTION

    # 3. 涡度辅助（单位：m/s）
    chunk['wind_vort_ms'] = chunk.apply(calculate_vorticity_based_wind, axis=1)

    # 转换为kt
    chunk['wind_vort'] = chunk['wind_vort_ms'] * MS_TO_KT

    # 4. 融合（主要用风速，涡度做修正）
    chunk['wind_fused'] = chunk['wind_10m'].copy()

    # 如果风速缺失但涡度存在，用涡度估算
    missing_mask = chunk['wind_fused'].isna() & chunk['wind_vort'].notna()
    chunk.loc[missing_mask, 'wind_fused'] = chunk.loc[missing_mask, 'wind_vort']

    # 如果涡度显示异常强/弱，微调风速
    both_exist = chunk['wind_10m'].notna() & chunk['wind_vort'].notna()
    vort_ratio = chunk.loc[both_exist, 'wind_vort'] / chunk.loc[both_exist, 'wind_10m']

    # 涡度比风速大很多 → 可能低估
    underestimate = both_exist & (vort_ratio > 1.3)
    chunk.loc[underestimate, 'wind_fused'] *= 1.1

    # 涡度比风速小很多 → 可能高估
    overestimate = both_exist & (vort_ratio < 0.7)
    chunk.loc[overestimate, 'wind_fused'] *= 0.9

    return chunk


def extract_typhoon_wind_smart(chunk):
    """
    智能提取台风风速（单位：kt）
    """

    # 计算增强风速
    chunk = calculate_enhanced_wind(chunk)

    chunk['distance_to_center'] = pd.to_numeric(chunk['distance_to_center'], errors='coerce')

    # 按台风时刻分组
    typhoon_winds = {}

    for (storm_id, time), group in chunk.groupby(['storm_id', 'time']):

        # 1. 排除风眼（中心10km内风速较小）
        outer_group = group[group['distance_to_center'] >= 10].copy()

        if len(outer_group) == 0:
            continue

        # 2. 寻找RMW范围内的最大风速
        rmw_group = outer_group[
            (outer_group['distance_to_center'] >= RMW_MIN) &
            (outer_group['distance_to_center'] <= RMW_MAX)
            ]

        if len(rmw_group) > 0:
            # RMW范围内的最大风速
            max_wind = rmw_group['wind_fused'].max()
        else:
            # 如果RMW范围没数据，扩大到150km
            extended_group = outer_group[outer_group['distance_to_center'] <= 150]
            if len(extended_group) > 0:
                max_wind = extended_group['wind_fused'].max()
            else:
                max_wind = outer_group['wind_fused'].max()

        key = (str(storm_id), str(time))
        typhoon_winds[key] = max_wind

    return typhoon_winds


def quality_control(wind_by_time):
    """
    质量控制（单位：kt）
    """

    print(f"\n应用质量控制...")

    # 转换为DataFrame便于处理
    wind_df = pd.DataFrame([
        {'storm_id': k[0], 'time': k[1], 'wind': v}
        for k, v in wind_by_time.items()
    ])

    original_count = len(wind_df)

    # 1. 合理性检查
    unreasonable = (wind_df['wind'] < MIN_REASONABLE_WIND) | (wind_df['wind'] > MAX_REASONABLE_WIND)
    n_unreasonable = unreasonable.sum()

    if n_unreasonable > 0:
        print(f"  发现 {n_unreasonable} 个不合理风速值")
        wind_df.loc[unreasonable, 'wind'] = np.nan

    # 2. 时间连续性检查（按台风分组）
    wind_df['time_dt'] = pd.to_datetime(wind_df['time'])
    wind_df = wind_df.sort_values(['storm_id', 'time_dt'])

    outliers = []

    for storm_id, storm_group in wind_df.groupby('storm_id'):
        storm_group = storm_group.sort_values('time_dt').copy()

        if len(storm_group) < 3:
            continue

        winds = storm_group['wind'].values
        times = storm_group['time_dt'].values

        for i in range(1, len(winds) - 1):
            if np.isnan(winds[i]):
                continue

            # 计算时间差（小时）
            dt1 = (times[i] - times[i - 1]).astype('timedelta64[h]').astype(float)
            dt2 = (times[i + 1] - times[i]).astype('timedelta64[h]').astype(float)

            # 计算变化率
            if dt1 > 0 and not np.isnan(winds[i - 1]):
                rate1 = abs(winds[i] - winds[i - 1]) / dt1
                if rate1 > MAX_HOURLY_CHANGE:
                    outliers.append(storm_group.index[i])

            if dt2 > 0 and not np.isnan(winds[i + 1]):
                rate2 = abs(winds[i + 1] - winds[i]) / dt2
                if rate2 > MAX_HOURLY_CHANGE:
                    outliers.append(storm_group.index[i])

    if len(outliers) > 0:
        outliers = list(set(outliers))
        print(f"  发现 {len(outliers)} 个时间异常点（变化过快）")
        wind_df.loc[outliers, 'wind'] = np.nan

    # 转回字典
    wind_by_time_qc = {}
    for _, row in wind_df.iterrows():
        if not np.isnan(row['wind']):
            key = (row['storm_id'], row['time'])
            wind_by_time_qc[key] = row['wind']

    removed = original_count - len(wind_by_time_qc)
    print(f"✓ 质量控制完成，移除 {removed} 个异常值")

    return wind_by_time_qc


def calculate_wind_from_era5_enhanced(input_csv, output_csv, ibtracs_dict, chunk_size):
    """
    增强版ERA5风速计算主函数（单位：kt）
    """

    print("\n" + "=" * 60)
    print("🌀 增强版ERA5台风风速计算（单位：kt）")
    print("=" * 60)

    print("\n增强特性:")
    print(f"  ✓ 多层风场融合（850/750/500/200hPa）")
    print(f"  ✓ 高度订正（850hPa → 10m，系数={HEIGHT_CORRECTION}）")
    print(f"  ✓ 台风结构优化（RMW={RMW_MIN}-{RMW_MAX}km）")
    print(f"  ✓ 涡度辅助校正")
    print(f"  ✓ 质量控制（合理性+连续性）")
    if ibtracs_dict:
        print(f"  ✓ 真实值对比验证")

    # 步骤1：检查列
    print("\n步骤1：检查数据列...")
    df_sample = pd.read_csv(input_csv, nrows=1000)

    required_cols = ['U_GRD_L100_850hPa', 'V_GRD_L100_850hPa', 'distance_to_center']
    missing_cols = [col for col in required_cols if col not in df_sample.columns]

    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False

    print(f"✓ 数据列检查通过")

    # 步骤2：计算风速
    print("\n步骤2：计算台风风速...")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    wind_by_time = {}
    total_rows = 0

    print("\n第一遍：提取台风最大风速...")

    with tqdm(desc="处理数据", unit=" 行", unit_scale=True) as pbar:
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):

            # 智能提取风速
            chunk_winds = extract_typhoon_wind_smart(chunk)

            # 合并到总字典
            for key, wind in chunk_winds.items():
                if key not in wind_by_time or wind > wind_by_time[key]:
                    wind_by_time[key] = wind

            total_rows += len(chunk)
            pbar.update(len(chunk))

    print(f"\n✓ 提取 {len(wind_by_time):,} 个台风时刻的风速")

    # 步骤3：质量控制
    wind_by_time = quality_control(wind_by_time)

    # 步骤4：与真实值对比（可选）
    if ibtracs_dict:
        compare_with_ibtracs(wind_by_time, ibtracs_dict)

    # 步骤5：统计
    wind_values = list(wind_by_time.values())
    print(f"\n风速统计（单位：kt）:")
    print(f"  最小: {min(wind_values):.1f} kt")
    print(f"  最大: {max(wind_values):.1f} kt")
    print(f"  平均: {np.mean(wind_values):.1f} kt")
    print(f"  中位数: {np.median(wind_values):.1f} kt")

    # 步骤6：写入文件
    print("\n第二遍：写入完整数据...")

    first_chunk = True
    calculated_rows = 0

    with tqdm(desc="写入文件", unit=" 行", unit_scale=True) as pbar:
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
            # 添加风速列（单位：kt），保持列名为typhoon_wind
            chunk['typhoon_wind'] = chunk.apply(
                lambda row: wind_by_time.get((str(row['storm_id']), str(row['time'])), np.nan),
                axis=1
            )

            calculated_rows += chunk['typhoon_wind'].notna().sum()

            # 写入
            chunk.to_csv(output_csv, mode='a', index=False, header=first_chunk)
            first_chunk = False

            pbar.update(len(chunk))

    print(f"\n✅ 计算完成")
    print(f"  总行数: {total_rows:,}")
    print(f"  有效风速: {calculated_rows:,} ({calculated_rows / total_rows * 100:.2f}%)")

    return True


def compare_with_ibtracs(wind_by_time, ibtracs_dict):
    """与IBTrACS真实值对比（单位：kt）"""

    print(f"\n对比真实值（IBTrACS）...")

    matched = []
    era5_winds = []
    true_winds = []

    for key, era5_wind in wind_by_time.items():
        if key in ibtracs_dict:
            true_wind = ibtracs_dict[key]

            if not np.isnan(true_wind) and not np.isnan(era5_wind):
                matched.append(key)
                era5_winds.append(era5_wind)
                true_winds.append(true_wind)

    if len(matched) == 0:
        print(f"  ⚠️ 未找到匹配的时刻")
        return

    era5_winds = np.array(era5_winds)
    true_winds = np.array(true_winds)

    # 计算误差
    bias = np.mean(era5_winds - true_winds)
    mae = np.mean(np.abs(era5_winds - true_winds))
    rmse = np.sqrt(np.mean((era5_winds - true_winds) ** 2))
    corr = np.corrcoef(era5_winds, true_winds)[0, 1]

    print(f"\n验证结果（{len(matched):,} 个匹配时刻）:")
    print(f"  偏差 (Bias):  {bias:+.1f} kt")
    print(f"  平均绝对误差 (MAE):  {mae:.1f} kt")
    print(f"  均方根误差 (RMSE): {rmse:.1f} kt")
    print(f"  相关系数 (Corr):  {corr:.3f}")

    # 分强度级别统计
    print(f"\n按强度分级验证:")

    levels = [
        ('热带风暴', 34, 48),
        ('强热带风暴', 48, 64),
        ('台风', 64, 85),
        ('强台风', 85, 100),
        ('超强台风', 100, 999)
    ]

    for level_name, v_min, v_max in levels:
        mask = (true_winds >= v_min) & (true_winds < v_max)
        if mask.sum() > 0:
            level_bias = np.mean(era5_winds[mask] - true_winds[mask])
            level_mae = np.mean(np.abs(era5_winds[mask] - true_winds[mask]))
            print(f"  {level_name:8s}: Bias={level_bias:+.1f} kt, MAE={level_mae:.1f} kt (n={mask.sum()})")


def verify_results_enhanced(output_csv):
    """增强验证（单位：kt）"""

    print("\n" + "=" * 60)
    print("步骤7：验证结果（单位：kt）")
    print("=" * 60)

    df = pd.read_csv(output_csv, nrows=100000)
    df['typhoon_wind'] = pd.to_numeric(df['typhoon_wind'], errors='coerce')

    total = len(df)
    valid = df['typhoon_wind'].notna().sum()

    print(f"\n数据统计（前 {len(df):,} 行）:")
    print(f"  有效值: {valid:,} ({valid / total * 100:.2f}%)")
    print(f"  缺失值: {total - valid:,} ({(total - valid) / total * 100:.2f}%)")

    if valid > 0:
        print(f"\n风速统计（单位：kt）:")
        print(f"  最小值: {df['typhoon_wind'].min():.1f} kt")
        print(f"  最大值: {df['typhoon_wind'].max():.1f} kt")
        print(f"  平均值: {df['typhoon_wind'].mean():.1f} kt")
        print(f"  中位数: {df['typhoon_wind'].median():.1f} kt")
        print(f"  标准差: {df['typhoon_wind'].std():.1f} kt")

        # 按等级统计
        print(f"\n按台风等级统计:")
        weak = df[df['typhoon_wind'] < 20].shape[0]
        td = df[(df['typhoon_wind'] >= 20) & (df['typhoon_wind'] < 34)].shape[0]
        ts = df[(df['typhoon_wind'] >= 34) & (df['typhoon_wind'] < 48)].shape[0]
        sts = df[(df['typhoon_wind'] >= 48) & (df['typhoon_wind'] < 64)].shape[0]
        ty = df[(df['typhoon_wind'] >= 64) & (df['typhoon_wind'] < 85)].shape[0]
        sty = df[(df['typhoon_wind'] >= 85) & (df['typhoon_wind'] < 100)].shape[0]
        ssty = df[df['typhoon_wind'] >= 100].shape[0]

        print(f"  <20 kt (弱):         {weak:6,} ({weak / valid * 100:5.2f}%)")
        print(f"  20-33 kt (热带低压): {td:6,} ({td / valid * 100:5.2f}%)")
        print(f"  34-47 kt (热带风暴): {ts:6,} ({ts / valid * 100:5.2f}%)")
        print(f"  48-63 kt (强热带风暴): {sts:6,} ({sts / valid * 100:5.2f}%)")
        print(f"  64-84 kt (台风):     {ty:6,} ({ty / valid * 100:5.2f}%)")
        print(f"  85-99 kt (强台风):   {sty:6,} ({sty / valid * 100:5.2f}%)")
        print(f"  ≥100 kt (超强台风):   {ssty:6,} ({ssty / valid * 100:5.2f}%)")

        # 示例数据
        print(f"\n风速最强的10个时刻:")
        sample_cols = ['storm_id', 'time', 'typhoon_name', 'typhoon_lat', 'typhoon_lon', 'typhoon_wind']
        sample_cols = [c for c in sample_cols if c in df.columns]
        top_wind = df.nlargest(10, 'typhoon_wind')[sample_cols]
        print(top_wind.to_string(index=False))


def main():
    import time
    start_time = time.time()

    try:
        # 加载IBTrACS（可选）
        ibtracs_dict = load_ibtracs_for_comparison(IBTRACS_CSV)

        # 计算风速
        success = calculate_wind_from_era5_enhanced(
            INPUT_CSV, OUTPUT_CSV, ibtracs_dict, CHUNK_SIZE
        )

        if not success:
            print("\n❌ 计算失败")
            return

        # 验证结果
        verify_results_enhanced(OUTPUT_CSV)

        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("🎉 全部完成！")
        print("=" * 60)
        print(f"总耗时: {elapsed:.1f} 秒 ({elapsed / 60:.1f} 分钟)")
        print(f"输出: {OUTPUT_CSV}")

        if os.path.exists(OUTPUT_CSV):
            size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
            print(f"文件大小: {size:.2f} MB")

        print(f"\n💡 增强特性总结:")
        print(f"  ✅ 多层风场融合（更全面）")
        print(f"  ✅ 高度订正（850hPa→10m，更准确）")
        print(f"  ✅ 台风结构优化（RMW，更合理）")
        print(f"  ✅ 涡度辅助（多变量，更鲁棒）")
        print(f"  ✅ 质量控制（异常检测，更可靠）")
        if ibtracs_dict:
            print(f"  ✅ 真实值验证（量化误差，更可信）")
        print(f"  ✅ 所有风速单位为节（kt），输出列名为typhoon_wind")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()