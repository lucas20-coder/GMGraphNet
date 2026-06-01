import os
import gc
import time
import logging
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================

NC_FILES = [
    r"f:/era5_output/surface_1965_1970.nc",
    r"f:/era5_output/surface_1971_1976.nc",
    r"f:/era5_output/surface_1977_1982.nc",
    r"f:/era5_output/surface_1983_1988.nc",
]

INPUT_CSV = r"f:/era5_output/typhoon_grid_7x7_era5_1965Q1-1986Q4.csv"
OUTPUT_CSV = r"f:/era5_output/typhoon_grid_7x7_era5_1965Q1-1986Q4_with_surface_FIXED.csv"

TIME_RANGE = ["1965-01-01", "1986-12-31"]

VARIABLE_MAPPING = {
    "t2m": "T2M",
    "2t": "T2M",
    "i10fg": "WIND_GUST_10M",
    "10fg": "WIND_GUST_10M",
    "fg10": "WIND_GUST_10M",
}

NEW_COLUMNS = ["T2M", "WIND_GUST_10M"]
INSERT_AFTER_COLUMN = "ABS_V_L100_850hPa"

CSV_CHUNK_SIZE = 300000  # 降低chunk大小，更好的内存控制

TIME_BATCH_SIZE = 1000
ENGINE = "netcdf4"


# ==================== 工具函数 ====================

def standardize_nc_dataset(ds):
    """标准化NC数据集"""
    rename_map = {}

    for tc in ['valid_time', 'TIME', 't']:
        if tc in ds.dims or tc in ds.coords:
            if tc != 'time':
                rename_map[tc] = 'time'
            break

    for lc in ['lat', 'LAT', 'y']:
        if lc in ds.dims or lc in ds.coords:
            if lc != 'latitude':
                rename_map[lc] = 'latitude'
            break

    for lc in ['lon', 'LON', 'x']:
        if lc in ds.dims or lc in ds.coords:
            if lc != 'longitude':
                rename_map[lc] = 'longitude'
            break

    if rename_map:
        ds = ds.rename(rename_map)

    if 'longitude' in ds.coords:
        lon_values = ds.longitude.values
        if lon_values.max() > 180:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby('longitude')

    return ds


def normalize_time_string(time_str):
    """
    🔥 关键修复：统一时间字符串格式
    确保merge_key的时间部分格式一致
    """
    try:
        # 尝试解析时间
        dt = pd.to_datetime(time_str, errors='coerce')
        if pd.isna(dt):
            return str(time_str)
        # 统一格式为 YYYY-MM-DD HH:MM:SS
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(time_str)


def scan_csv_for_unique_points(csv_file, chunk_size=1200000):
    """扫描CSV收集唯一时空点（带时间格式标准化）"""
    logger.info(f"\n{'=' * 60}")
    logger.info("步骤1：扫描CSV收集唯一时空点")
    logger.info(f"{'=' * 60}")

    unique_points = set()
    total_rows = 0
    time_format_samples = []

    with tqdm(desc="扫描CSV", unit=" 行", unit_scale=True, ncols=100) as pbar:
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False,
                                 usecols=['time', 'lat', 'lon']):

            chunk['lat'] = chunk['lat'].round(2)
            chunk['lon'] = chunk['lon'].round(2)

            # 🔥 标准化时间格式
            chunk['time_normalized'] = chunk['time'].apply(normalize_time_string)

            # 收集时间格式样例
            if len(time_format_samples) < 10:
                for orig, norm in zip(chunk['time'].head(10), chunk['time_normalized'].head(10)):
                    time_format_samples.append((orig, norm))

            for _, row in chunk.iterrows():
                point = (row['time_normalized'], row['lat'], row['lon'])
                unique_points.add(point)

            total_rows += len(chunk)
            pbar.update(len(chunk))
            pbar.set_postfix({
                '唯一点': f"{len(unique_points):,}",
                '压缩率': f"{len(unique_points) / total_rows * 100:.1f}%"
            })

    logger.info(f"\n✓ 扫描完成")
    logger.info(f"  总行数: {total_rows:,}")
    logger.info(f"  唯一时空点: {len(unique_points):,}")
    logger.info(f"  压缩率: {len(unique_points) / total_rows * 100:.2f}%")

    # 显示时间格式转换样例
    logger.info(f"\n时间格式标准化样例:")
    for orig, norm in time_format_samples[:5]:
        logger.info(f"  {orig} -> {norm}")

    return unique_points


def extract_nc_data_batch(nc_files, unique_points, time_range):
    """
    提取NC数据（保持原有逻辑，但time使用标准化格式）
    """
    import psutil

    def mem_mb():
        return psutil.Process().memory_info().rss / (1024 ** 2)

    def build_time_index_map(ds_time_index, query_times, tolerance_hours=3):
        if len(ds_time_index) == 0:
            return {}
        ds_ns = ds_time_index.view("i8")
        q_ns = query_times.view("i8")
        pos = np.searchsorted(ds_ns, q_ns)
        pos = np.clip(pos, 1, len(ds_ns) - 1)
        left_idx = pos - 1
        right_idx = pos
        left_diff = np.abs(q_ns - ds_ns[left_idx])
        right_diff = np.abs(ds_ns[right_idx] - q_ns)
        choose_right = right_diff < left_diff
        idx = np.where(choose_right, right_idx, left_idx)
        tol_ns = tolerance_hours * 3600 * 1_000_000_000
        good = np.minimum(left_diff, right_diff) <= tol_ns
        mapping = {}
        for t, ok, i in zip(query_times, good, idx):
            if ok:
                mapping[pd.Timestamp(t)] = int(i)
        return mapping

    def to_nearest_indices(axis_vals, query_vals, descending=False):
        vals = axis_vals[::-1] if descending else axis_vals
        pos = np.searchsorted(vals, query_vals)
        pos = np.clip(pos, 1, len(vals) - 1)
        left = vals[pos - 1]
        right = vals[pos]
        choose_right = (np.abs(right - query_vals) < np.abs(query_vals - left))
        idx = np.where(choose_right, pos, pos - 1)
        if descending:
            idx = (len(vals) - 1) - idx
        return idx.astype(int)

    logger.info(f"\n{'=' * 60}")
    logger.info("步骤2：提取NC数据")
    logger.info(f"{'=' * 60}")

    logger.info("\n准备数据...")
    df_points = pd.DataFrame(list(unique_points), columns=['time', 'lat', 'lon'])
    df_points['time_obj'] = pd.to_datetime(df_points['time'], format='mixed', errors='coerce')
    bad = df_points['time_obj'].isna().sum()
    if bad > 0:
        logger.warning(f"  ⚠️ 发现 {bad} 个无效时间，已丢弃")
        df_points = df_points.dropna(subset=['time_obj'])

    df_points['lat'] = df_points['lat'].round(2)
    df_points['lon'] = df_points['lon'].round(2)
    logger.info(f"  ✓ 唯一点数: {len(df_points):,}")
    logger.info(f"  ✓ 唯一时间: {df_points['time_obj'].nunique():,}")
    logger.info(f"  ✓ 时间范围: {df_points['time_obj'].min()} ~ {df_points['time_obj'].max()}")

    logger.info("\n打开NC文件...")
    datasets = []
    var_mapping = None

    for i, nc_file in enumerate(nc_files, 1):
        logger.info(f"[{i}/{len(nc_files)}] {os.path.basename(nc_file)}")
        ds = xr.open_dataset(nc_file, engine=ENGINE)
        ds = standardize_nc_dataset(ds)
        if time_range:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))

        if 'time' not in ds.dims and 'time' not in ds.coords:
            logger.warning("  ⚠️ 未找到 time 维，跳过")
            ds.close()
            continue

        tmin = pd.Timestamp(ds.time.values.min())
        tmax = pd.Timestamp(ds.time.values.max())
        logger.info(f"  ✓ 时间范围: {tmin} ~ {tmax}")
        logger.info(f"  ✓ 时间点数: {len(ds.time):,}")

        if var_mapping is None:
            var_mapping = {}
            for v in ds.data_vars:
                for key, std in VARIABLE_MAPPING.items():
                    if v.lower() == key.lower() and std not in var_mapping:
                        var_mapping[std] = v
                        logger.info(f"  ✓ {v} -> {std}")

        wanted = [v for v in (var_mapping or {}).values() if v in ds.data_vars]
        if wanted:
            ds = ds[wanted]

        time_index = pd.DatetimeIndex(ds.time.values)

        datasets.append({
            'ds': ds,
            'time_min': tmin,
            'time_max': tmax,
            'time_index': time_index,
            'file': nc_file
        })

    if not datasets:
        logger.error("✗ 无可用数据集")
        return None

    logger.info("\n路由时间点到数据集...")
    time_to_dsidx = {}
    unique_times = df_points['time_obj'].unique()
    for t in unique_times:
        for idx, info in enumerate(datasets):
            if info['time_min'] <= t <= info['time_max']:
                time_to_dsidx[t] = idx
                break

    df_points['dataset_idx'] = df_points['time_obj'].map(time_to_dsidx)
    before = len(df_points)
    df_points = df_points.dropna(subset=['dataset_idx'])
    df_points['dataset_idx'] = df_points['dataset_idx'].astype(int)
    after = len(df_points)
    if before > after:
        logger.warning(f"  ⚠️ 有 {before - after} 个点未命中数据集")

    logger.info("\n开始提取...")
    results = []

    for ds_idx, info in enumerate(datasets):
        logger.info(f"\n[{ds_idx + 1}/{len(datasets)}] {os.path.basename(info['file'])}")

        pts = df_points[df_points['dataset_idx'] == ds_idx].copy()
        if len(pts) == 0:
            logger.info("  跳过（无点）")
            continue

        lat_min, lat_max = float(pts['lat'].min()), float(pts['lat'].max())
        lon_min, lon_max = float(pts['lon'].min()), float(pts['lon'].max())
        lat_pad, lon_pad = 0.5, 0.5

        ds = info['ds']
        lat_vals = ds.latitude.values
        lat_desc = bool(lat_vals[0] > lat_vals[-1])
        if lat_desc:
            ds = ds.sel(latitude=slice(min(lat_max + lat_pad, lat_vals[0]),
                                       max(lat_min - lat_pad, lat_vals[-1])))
        else:
            ds = ds.sel(latitude=slice(max(lat_min - lat_pad, lat_vals[0]),
                                       min(lat_max + lat_pad, lat_vals[-1])))

        lon_vals = ds.longitude.values
        lo0, loN = float(lon_vals[0]), float(lon_vals[-1])
        if lon_min <= lon_max:
            ds = ds.sel(longitude=slice(max(lon_min - lon_pad, min(lo0, loN)),
                                        min(lon_max + lon_pad, max(lo0, loN))))

        lat_vals = ds.latitude.values
        lon_vals = ds.longitude.values
        lat_desc = bool(lat_vals[0] > lat_vals[-1])

        logger.info(f"  数据点: {len(pts):,}")

        unique_times_ds = pd.DatetimeIndex(pts['time_obj'].unique())
        tmap = build_time_index_map(info['time_index'], unique_times_ds)
        if not tmap:
            logger.warning("  ⚠️ 无匹配时间")
            continue

        pts = pts[pts['time_obj'].isin(tmap.keys())].copy()
        pts['time_idx'] = pts['time_obj'].map(tmap).astype(int)

        grouped = pts.groupby('time_idx', sort=True)
        time_indices = list(grouped.groups.keys())

        with tqdm(total=len(time_indices), desc=f"  数据集{ds_idx + 1}", unit="时间", ncols=100) as pbar:
            for ti in time_indices:
                try:
                    data_at_time = ds.isel(time=int(ti))
                    pts_t = grouped.get_group(ti)

                    lats = pts_t['lat'].values.astype(float)
                    lons = pts_t['lon'].values.astype(float)

                    lat_idx = to_nearest_indices(lat_vals, lats, descending=lat_desc)
                    lon_idx = to_nearest_indices(lon_vals, lons, descending=False)

                    extracted = data_at_time.isel(
                        latitude=xr.DataArray(lat_idx, dims='points'),
                        longitude=xr.DataArray(lon_idx, dims='points')
                    )

                    for j, (_, row) in enumerate(pts_t.iterrows()):
                        # 🔥 保持标准化的时间格式
                        rec = {'time': row['time'], 'lat': row['lat'], 'lon': row['lon']}
                        if var_mapping:
                            for std, ncname in var_mapping.items():
                                if ncname in extracted.data_vars:
                                    try:
                                        rec[std] = float(extracted[ncname].isel(points=j).values)
                                    except:
                                        rec[std] = np.nan
                                else:
                                    rec[std] = np.nan
                        results.append(rec)

                except Exception as e:
                    logger.debug(f"    时间索引 {ti} 失败: {e}")
                finally:
                    pbar.update(1)

        del ds
        gc.collect()

    logger.info("\n关闭数据集...")
    for info in datasets:
        try:
            info['ds'].close()
        except:
            pass
    gc.collect()

    if not results:
        logger.error("✗ 未提取到数据")
        return None

    df = pd.DataFrame(results)
    logger.info(f"\n✅ 提取完成: {len(df):,} 条")

    return df


def merge_with_extracted_data(input_csv, extracted_df, output_csv, chunk_size):
    """
    🔥 关键修复：合并时保护time列
    """
    logger.info(f"\n{'=' * 60}")
    logger.info("步骤3：安全合并（保护time列）")
    logger.info(f"{'=' * 60}")

    logger.info(f"\n准备提取数据...")
    extracted_df['lat'] = extracted_df['lat'].round(2)
    extracted_df['lon'] = extracted_df['lon'].round(2)

    # 🔥 确保时间格式一致
    extracted_df['merge_key'] = (
            extracted_df['time'].astype(str) + '_' +
            extracted_df['lat'].astype(str) + '_' +
            extracted_df['lon'].astype(str)
    )

    merge_columns = ['merge_key'] + NEW_COLUMNS
    extracted_for_merge = extracted_df[merge_columns].copy()

    logger.info(f"✓ 准备完成: {len(extracted_for_merge):,} 条记录")
    logger.info(f"  样例merge_key: {list(extracted_for_merge['merge_key'].head(3))}")

    sample = pd.read_csv(input_csv, nrows=5)

    if INSERT_AFTER_COLUMN not in sample.columns:
        logger.error(f"✗ 未找到列: {INSERT_AFTER_COLUMN}")
        return False

    insert_idx = list(sample.columns).index(INSERT_AFTER_COLUMN) + 1
    old_columns = list(sample.columns)
    new_column_order = old_columns[:insert_idx] + NEW_COLUMNS + old_columns[insert_idx:]

    logger.info(f"列插入位置: 第 {insert_idx} 列")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    logger.info(f"\n开始安全合并...")

    first_chunk = True
    total_rows = 0
    matched_count = 0
    merge_key_match_count = 0
    time_preserved_count = 0

    with tqdm(desc="合并CSV", unit=" 行", unit_scale=True, ncols=100) as pbar:
        for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):

            # 🔥 保存原始time列
            original_time = chunk['time'].copy()
            original_row_count = len(chunk)

            chunk['lat'] = chunk['lat'].round(2)
            chunk['lon'] = chunk['lon'].round(2)

            # 🔥 标准化时间格式用于merge_key
            chunk['time_normalized'] = chunk['time'].apply(normalize_time_string)

            chunk['merge_key'] = (
                    chunk['time_normalized'].astype(str) + '_' +
                    chunk['lat'].astype(str) + '_' +
                    chunk['lon'].astype(str)
            )

            # merge前统计有多少merge_key能匹配
            before_merge = chunk['merge_key'].isin(extracted_for_merge['merge_key']).sum()
            merge_key_match_count += before_merge

            chunk_merged = chunk.merge(
                extracted_for_merge,
                on='merge_key',
                how='left',
                suffixes=('', '_extracted')
            )

            # 🔥 验证merge没有改变行数
            if len(chunk_merged) != original_row_count:
                logger.warning(f"  ⚠️ merge改变了行数: {original_row_count} -> {len(chunk_merged)}")

            # 🔥 恢复原始time列（确保不被覆盖）
            chunk_merged['time'] = original_time.values
            time_preserved_count += chunk_merged['time'].notna().sum()

            chunk_merged = chunk_merged.drop(columns=['merge_key', 'time_normalized'], errors='ignore')

            for col in NEW_COLUMNS:
                if col in chunk_merged.columns:
                    matched_count += chunk_merged[col].notna().sum()

            chunk_merged = chunk_merged[new_column_order]

            chunk_merged.to_csv(output_csv, mode='a', index=False, header=first_chunk)
            first_chunk = False

            total_rows += len(chunk)
            match_rate = matched_count / (total_rows * len(NEW_COLUMNS)) * 100 if total_rows > 0 else 0

            pbar.update(len(chunk))
            pbar.set_postfix({
                '匹配率': f"{match_rate:.1f}%",
                'time完整': f"{time_preserved_count / total_rows * 100:.1f}%"
            })

            del chunk, chunk_merged, original_time
            gc.collect()

    logger.info(f"\n✅ 合并完成!")
    logger.info(f"  总行数: {total_rows:,}")
    logger.info(
        f"  time列保留: {time_preserved_count:,}/{total_rows:,} ({time_preserved_count / total_rows * 100:.2f}%)")
    logger.info(
        f"  merge_key匹配: {merge_key_match_count:,}/{total_rows:,} ({merge_key_match_count / total_rows * 100:.2f}%)")
    logger.info(
        f"  变量数据点匹配: {matched_count:,}/{total_rows * len(NEW_COLUMNS):,} ({matched_count / (total_rows * len(NEW_COLUMNS)) * 100:.2f}%)")

    # 🔥 验证输出文件
    logger.info(f"\n验证输出文件...")
    verify_total = 0
    verify_time_valid = 0
    verify_empty_strings = 0

    for chunk in pd.read_csv(output_csv, chunksize=chunk_size, low_memory=False, keep_default_na=False):
        verify_total += len(chunk)
        chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
        verify_time_valid += chunk['time'].notna().sum()

        for col in NEW_COLUMNS:
            if col in chunk.columns:
                verify_empty_strings += chunk[col].apply(lambda x: str(x).strip() == "").sum()

    logger.info(f"  验证总行数: {verify_total:,}")
    logger.info(f"  time列有效: {verify_time_valid:,}/{verify_total:,} ({verify_time_valid / verify_total * 100:.2f}%)")
    logger.info(f"  空字符串数: {verify_empty_strings:,}")

    if verify_time_valid < total_rows * 0.99:
        logger.error(f"  ⚠️ time列完整性低于99%！")
        return False

    if verify_empty_strings > 0:
        logger.warning(f"  ⚠️ 仍有 {verify_empty_strings:,} 个空字符串")

    return True


def main():
    """主函数"""
    logger.info("\n" + "=" * 60)
    logger.info("⚡ 修复版：保护time列完整性")
    logger.info("=" * 60)

    for nc_file in NC_FILES:
        if not os.path.exists(nc_file):
            logger.error(f"✗ 文件不存在: {nc_file}")
            return

    if not os.path.exists(INPUT_CSV):
        logger.error(f"✗ CSV不存在: {INPUT_CSV}")
        return

    overall_start = time.time()

    try:
        unique_points = scan_csv_for_unique_points(INPUT_CSV, CSV_CHUNK_SIZE)
        if unique_points is None:
            return

        extracted_df = extract_nc_data_batch(NC_FILES, unique_points, TIME_RANGE)
        if extracted_df is None:
            return

        success = merge_with_extracted_data(INPUT_CSV, extracted_df, OUTPUT_CSV, CSV_CHUNK_SIZE)

        if not success:
            return

        total_elapsed = time.time() - overall_start
        logger.info(f"\n{'=' * 60}")
        logger.info("🎉 全部完成!")
        logger.info(f"{'=' * 60}")
        logger.info(f"总耗时: {total_elapsed:.1f} 秒 ({total_elapsed / 60:.1f} 分钟)")
        logger.info(f"输出: {OUTPUT_CSV}")

        if os.path.exists(OUTPUT_CSV):
            size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
            logger.info(f"文件大小: {size:.2f} MB")

    except Exception as e:
        logger.error(f"✗ 错误: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"开始时间: {start_time}\n")

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 用户中断")
    except Exception as e:
        logger.error(f"\n✗ 错误: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        end_time = datetime.now()
        logger.info(f"\n结束时间: {end_time}")
        logger.info(f"总耗时: {end_time - start_time}")