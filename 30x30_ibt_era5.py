import os
import gc
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import psutil
from scipy.spatial import cKDTree
from multiprocessing import Pool, Manager, cpu_count
import threading
import time
import re

warnings.filterwarnings("ignore")


def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_system_memory_info():
    """获取系统内存信息"""
    mem = psutil.virtual_memory()
    return {
        'total': mem.total / 1024 / 1024 / 1024,
        'available': mem.available / 1024 / 1024 / 1024,
        'percent': mem.percent,
        'used': mem.used / 1024 / 1024 / 1024
    }


class StreamingProgressTracker:
    """流式处理进度跟踪器"""

    def __init__(self, total_estimated_chunks):
        self.total_estimated_chunks = total_estimated_chunks
        self.processed_chunks = 0
        self.matched_typhoons = 0
        self.generated_grids = 0
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()

    def update_progress(self, chunk_matched=0, chunk_grids=0):
        """更新进度"""
        self.processed_chunks += 1
        self.matched_typhoons += chunk_matched
        self.generated_grids += chunk_grids

        current_time = datetime.now()
        elapsed = current_time - self.start_time

        # 计算速度
        if elapsed.total_seconds() > 0:
            chunk_rate = self.processed_chunks / elapsed.total_seconds()
            grid_rate = self.generated_grids / elapsed.total_seconds()
        else:
            chunk_rate = grid_rate = 0

        # 估算剩余时间
        if self.processed_chunks > 0:
            remaining_chunks = max(0, self.total_estimated_chunks - self.processed_chunks)
            if chunk_rate > 0:
                eta = remaining_chunks / chunk_rate
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "计算中"
        else:
            eta_str = "计算中"

        progress_pct = (self.processed_chunks / self.total_estimated_chunks) * 100

        return {
            'processed_chunks': self.processed_chunks,
            'total_chunks': self.total_estimated_chunks,
            'progress_pct': progress_pct,
            'matched_typhoons': self.matched_typhoons,
            'generated_grids': self.generated_grids,
            'elapsed': str(elapsed).split('.')[0],
            'eta': eta_str,
            'chunk_rate': chunk_rate,
            'grid_rate': grid_rate
        }


def standardize_column_names(df):
    """
    🔥 标准化ERA5列名：统一气压层级格式

    将 500.0hPa, 850.0hPa 等统一为 500hPa, 850hPa
    """
    new_columns = {}

    for col in df.columns:
        # 匹配气压层级模式：xxx.0hPa -> xxxhPa
        standardized_col = re.sub(r'(\d+)\.0+hPa', r'\1hPa', col)
        # 也处理可能的其他浮点数格式
        standardized_col = re.sub(r'(\d+)\.0+_', r'\1_', standardized_col)

        if standardized_col != col:
            new_columns[col] = standardized_col

    if new_columns:
        df = df.rename(columns=new_columns)

        # 首次调用时打印转换信息
        if not hasattr(standardize_column_names, '_first_call_done'):
            print(f"\n🔧 列名标准化：")
            print(f"   转换了 {len(new_columns)} 个列名")
            print(f"   示例转换:")
            for old, new in list(new_columns.items())[:5]:
                print(f"      {old} -> {new}")
            standardize_column_names._first_call_done = True

    return df


def find_era5_quarterly_files(data_dirs, start_year, start_quarter, end_year, end_quarter):
    """在多个目录中查找ERA5季度文件并按时间顺序排序"""
    print("=== 查找ERA5季度文件 ===")

    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    print(f"搜索目录数: {len(data_dirs)}")
    for idx, dir_path in enumerate(data_dirs, 1):
        print(f"  目录 {idx}: {dir_path}")
    print()

    all_files = []

    # 遍历年份和季度
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            if year == start_year and quarter < start_quarter:
                continue
            if year == end_year and quarter > end_quarter:
                continue

            filename = f"era5_pl_{year}_Q{quarter}.csv"

            file_found = False
            for data_dir in data_dirs:
                filepath = os.path.join(data_dir, filename)

                if os.path.exists(filepath):
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    all_files.append({
                        'path': filepath,
                        'year': year,
                        'quarter': quarter,
                        'size_mb': file_size_mb,
                        'source_dir': data_dir
                    })
                    print(f"  ✓ 找到: {filename} ({file_size_mb:.2f} MB) - 来源: {os.path.basename(data_dir)}")
                    file_found = True
                    break

            if not file_found:
                print(f"  ✗ 缺失: {filename}")

    print(f"\n总共找到 {len(all_files)} 个季度文件")

    if all_files:
        from collections import Counter
        dir_counts = Counter(f['source_dir'] for f in all_files)
        print("\n各目录文件分布:")
        for dir_path, count in dir_counts.items():
            print(f"  {os.path.basename(dir_path)}: {count} 个文件")

    return all_files


def load_ibtracs_data(ibtracs_file):
    """加载IBTrACS数据到内存"""
    print("=== 加载IBTrACS数据 ===")

    try:
        ibtracs = pd.read_csv(ibtracs_file)
        print(f"原始IBTrACS数据: {len(ibtracs):,} 行")

        # 时间转换
        if 'ISO_TIME' in ibtracs.columns:
            ibtracs['time'] = pd.to_datetime(ibtracs['ISO_TIME'])
        else:
            ibtracs['time'] = pd.to_datetime(ibtracs['time'])

        # 移除时区信息
        if ibtracs['time'].dt.tz is not None:
            ibtracs['time'] = ibtracs['time'].dt.tz_localize(None)
            print("✓ 已移除 IBTrACS 时间的时区信息")

        # 列名映射
        column_mappings = {
            'typhoon_name': ['NAME', 'STORM_NAME'],
            'typhoon_lat': ['LAT', 'LATITUDE'],
            'typhoon_lon': ['LON', 'LONGITUDE'],
            'typhoon_wind': ['WMO_WIND', 'WIND', 'MAX_WIND', 'USA_WIND']
        }

        for target_col, possible_cols in column_mappings.items():
            found_col = None
            for col in possible_cols:
                if col in ibtracs.columns:
                    found_col = col
                    break
            if found_col:
                ibtracs[target_col] = ibtracs[found_col]
            else:
                print(f"Warning: 未找到 {target_col} 对应的列")
                ibtracs[target_col] = np.nan

        # 生成storm_id
        if 'SID' in ibtracs.columns:
            ibtracs['storm_id'] = ibtracs['SID']
        else:
            ibtracs['storm_id'] = ibtracs['typhoon_name'] + "_" + ibtracs['time'].dt.year.astype(str)

        # 清理数据
        ibtracs = ibtracs.dropna(subset=['typhoon_lat', 'typhoon_lon', 'time'])

        print(f"处理后IBTrACS数据: {len(ibtracs):,} 行")
        print(f"时间范围: {ibtracs['time'].min()} 到 {ibtracs['time'].max()}")

        return ibtracs

    except Exception as e:
        print(f"❌ 加载IBTrACS数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_typhoon_grid_28x28_0p25deg(era5_subset, typhoon_lat, typhoon_lon):
    """
    🔥 生成28×28台风网格（0.25°分辨率）- 使用真实ERA5网格点经纬度
    """
    # ========== 核心参数配置 ==========
    GRID_SIZE = 50
    RESOLUTION = 0.25
    HALF_SIZE = (GRID_SIZE - 1) / 2.0
    SPATIAL_RANGE = HALF_SIZE * RESOLUTION
    SEARCH_RADIUS = SPATIAL_RANGE * 1.5

    # 地理预筛选
    geo_mask = (
            (era5_subset['lat'] >= typhoon_lat - SEARCH_RADIUS) &
            (era5_subset['lat'] <= typhoon_lat + SEARCH_RADIUS) &
            (era5_subset['lon'] >= typhoon_lon - SEARCH_RADIUS) &
            (era5_subset['lon'] <= typhoon_lon + SEARCH_RADIUS)
    )

    nearby_era5 = era5_subset[geo_mask]

    if len(nearby_era5) == 0:
        return []

    # 生成理论网格（仅用于匹配）
    lat_start = typhoon_lat - SPATIAL_RANGE
    lon_start = typhoon_lon - SPATIAL_RANGE

    grid_lats_theory = np.arange(lat_start, lat_start + GRID_SIZE * RESOLUTION, RESOLUTION)
    grid_lons_theory = np.arange(lon_start, lon_start + GRID_SIZE * RESOLUTION, RESOLUTION)

    grid_lats_theory = np.round(grid_lats_theory, 2)[:GRID_SIZE]
    grid_lons_theory = np.round(grid_lons_theory, 2)[:GRID_SIZE]

    # KDTree查找
    era5_coords = np.column_stack([nearby_era5['lat'].values, nearby_era5['lon'].values])
    tree = cKDTree(era5_coords)

    exclude_cols = ['time', 'lat', 'lon', 'latitude', 'longitude', 'time_rounded']
    available_vars = [col for col in nearby_era5.columns if col not in exclude_cols]

    grid_data = []

    for i, grid_lat_theory in enumerate(grid_lats_theory):
        for j, grid_lon_theory in enumerate(grid_lons_theory):
            # 使用理论坐标查找最近的ERA5点
            distance, idx = tree.query([grid_lat_theory, grid_lon_theory])
            nearest_row = nearby_era5.iloc[idx]

            # 🔥 关键修改：使用真实ERA5网格点的经纬度
            actual_lat = nearest_row['lat']
            actual_lon = nearest_row['lon']

            # 计算到台风中心的距离（使用真实坐标）
            distance_to_center = np.sqrt((actual_lat - typhoon_lat) ** 2 +
                                         (actual_lon - typhoon_lon) ** 2) * 111.32

            grid_point = {
                'grid_i': i,
                'grid_j': j,
                'grid_lat': round(actual_lat, 2),  # 真实经度
                'grid_lon': round(actual_lon, 2),  # 真实纬度
                'distance_to_center': round(distance_to_center, 2),
                'lat': round(actual_lat, 2),  # 真实纬度（兼容性）
                'lon': round(actual_lon, 2),  # 真实经度（兼容性）
            }

            for var in available_vars:
                try:
                    grid_point[var] = nearest_row[var]
                except:
                    grid_point[var] = np.nan

            grid_data.append(grid_point)

    return grid_data


def process_era5_chunk_with_ibtracs(era5_chunk, ibtracs_data, max_time_diff_hours=1.5):
    """处理单个ERA5数据块并与IBTrACS匹配"""
    # 预处理时间
    if 'time' in era5_chunk.columns and era5_chunk['time'].dtype == 'object':
        era5_chunk['time'] = era5_chunk['time'].str.replace(r' (\d{2})-(\d{2})$', r' \1:\2', regex=True)

    era5_chunk['time'] = pd.to_datetime(era5_chunk['time'], format='mixed', errors='coerce')

    if hasattr(era5_chunk['time'], 'dt') and era5_chunk['time'].dt.tz is not None:
        era5_chunk['time'] = era5_chunk['time'].dt.tz_localize(None)

    era5_chunk = era5_chunk.dropna(subset=['time', 'lat', 'lon'])

    if len(era5_chunk) == 0:
        return pd.DataFrame(), 0

    # 标准化列名
    era5_chunk = standardize_column_names(era5_chunk)

    # 构建时间索引
    era5_chunk['time_rounded'] = era5_chunk['time'].dt.round('H')

    era5_time_dict = {}
    for time_val, group_df in era5_chunk.groupby('time_rounded'):
        era5_time_dict[time_val] = group_df

    era5_times = sorted(era5_time_dict.keys())

    if len(era5_times) == 0:
        return pd.DataFrame(), 0

    # 找到时间窗口
    time_margin = pd.Timedelta(hours=max_time_diff_hours + 1)
    chunk_time_min = min(era5_times) - time_margin
    chunk_time_max = max(era5_times) + time_margin

    ibtracs_window = ibtracs_data[
        (ibtracs_data['time'] >= chunk_time_min) &
        (ibtracs_data['time'] <= chunk_time_max)
        ].copy()

    if len(ibtracs_window) == 0:
        return pd.DataFrame(), 0

    # 匹配处理
    all_grid_data = []
    matched_count = 0

    era5_times_array = np.array([t.timestamp() for t in era5_times])

    for _, typhoon_row in ibtracs_window.iterrows():
        typhoon_time = typhoon_row['time']
        typhoon_time_rounded = typhoon_time.round('H')
        typhoon_timestamp = typhoon_time_rounded.timestamp()

        time_diffs = np.abs(era5_times_array - typhoon_timestamp) / 3600
        min_time_diff = np.min(time_diffs)

        if min_time_diff <= max_time_diff_hours:
            closest_idx = np.argmin(time_diffs)
            closest_era5_time = era5_times[closest_idx]

            if closest_era5_time in era5_time_dict:
                era5_subset = era5_time_dict[closest_era5_time]

                grid_points = create_typhoon_grid_28x28_0p25deg(
                    era5_subset,
                    typhoon_row['typhoon_lat'],
                    typhoon_row['typhoon_lon']
                )

                expected_size = 50 * 50
                if len(grid_points) == expected_size:
                    matched_count += 1

                    for point in grid_points:
                        point.update({
                            'typhoon_name': typhoon_row['typhoon_name'],
                            'typhoon_lat': typhoon_row['typhoon_lat'],
                            'typhoon_lon': typhoon_row['typhoon_lon'],
                            'typhoon_wind': typhoon_row['typhoon_wind'],
                            'storm_id': typhoon_row['storm_id'],
                            'time': typhoon_time,
                            'era5_time': closest_era5_time,
                            'time_diff_hours': min_time_diff,
                        })

                    all_grid_data.extend(grid_points)

    if all_grid_data:
        return pd.DataFrame(all_grid_data), matched_count
    else:
        return pd.DataFrame(), 0


def get_ibtracs_for_quarter(ibtracs_full, year, quarter, time_margin_days=5):
    """根据季度提取对应的IBTrACS数据子集"""
    quarter_start_month = (quarter - 1) * 3 + 1
    quarter_start = pd.Timestamp(year=year, month=quarter_start_month, day=1)

    if quarter == 4:
        quarter_end = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
    else:
        quarter_end = pd.Timestamp(year=year, month=quarter_start_month + 3, day=1) - pd.Timedelta(days=1)

    time_margin = pd.Timedelta(days=time_margin_days)
    filter_start = quarter_start - time_margin
    filter_end = quarter_end + time_margin

    ibtracs_subset = ibtracs_full[
        (ibtracs_full['time'] >= filter_start) &
        (ibtracs_full['time'] <= filter_end)
        ].copy()

    return ibtracs_subset, filter_start, filter_end


def process_single_quarterly_file(args):
    """
    🔥 处理单个季度文件（修复进度同步BUG）
    """
    file_info, ibtracs_quarter_data, temp_output_dir, chunk_size, progress_dict, worker_id = args

    file_path = file_info['path']
    year = file_info['year']
    quarter = file_info['quarter']

    temp_output_file = os.path.join(
        temp_output_dir,
        f"temp_{year}Q{quarter}_{os.getpid()}.csv"
    )

    process_start_time = datetime.now()

    try:
        # 🔥 修复：完整赋值才能同步
        progress_dict[worker_id] = {
            'year': year,
            'quarter': quarter,
            'status': '处理中',
            'chunk_count': 0,
            'matched_typhoons': 0,
            'generated_grids': 0,
            'start_time': process_start_time.isoformat(),
            'last_update': datetime.now().isoformat()
        }

        # 检查该季度是否有台风数据
        if len(ibtracs_quarter_data) == 0:
            progress_dict[worker_id] = {
                'year': year,
                'quarter': quarter,
                'status': '完成(无数据)',
                'chunk_count': 0,
                'matched_typhoons': 0,
                'generated_grids': 0,
                'start_time': process_start_time.isoformat(),
                'last_update': datetime.now().isoformat()
            }
            return {
                'year': year,
                'quarter': quarter,
                'success': True,
                'matched_typhoons': 0,
                'generated_grids': 0,
                'processed_chunks': 0,
                'temp_file': None,
                'elapsed_time': (datetime.now() - process_start_time).total_seconds(),
                'error': None
            }

        # 转换为DataFrame
        if isinstance(ibtracs_quarter_data, dict):
            ibtracs_quarter = pd.DataFrame(ibtracs_quarter_data)
        else:
            ibtracs_quarter = ibtracs_quarter_data

        write_buffer = []
        buffer_size_limit = 12
        first_chunk = True

        total_matched = 0
        total_grids = 0
        chunk_count = 0

        # 分块处理ERA5文件
        for era5_chunk in pd.read_csv(
                file_path,
                chunksize=chunk_size,
                on_bad_lines='skip',
                engine='python'
        ):
            chunk_count += 1

            # 处理chunk
            chunk_result, chunk_matched = process_era5_chunk_with_ibtracs(
                era5_chunk, ibtracs_quarter,
                max_time_diff_hours=1.5
            )

            if not chunk_result.empty:
                # 格式化时间列
                if 'time' in chunk_result.columns:
                    chunk_result['time'] = pd.to_datetime(chunk_result['time']).dt.strftime('%Y/%m/%d %H:%M')
                if 'era5_time' in chunk_result.columns:
                    chunk_result['era5_time'] = pd.to_datetime(chunk_result['era5_time']).dt.strftime('%Y/%m/%d %H:%M')

                write_buffer.append(chunk_result)

            total_matched += chunk_matched
            total_grids += len(chunk_result)

            # 🔥 修复：每次都完整赋值（不能用.update()）
            progress_dict[worker_id] = {
                'year': year,
                'quarter': quarter,
                'status': '处理中',
                'chunk_count': chunk_count,
                'matched_typhoons': total_matched,
                'generated_grids': total_grids,
                'start_time': process_start_time.isoformat(),
                'last_update': datetime.now().isoformat()
            }

            # 批量写入
            if len(write_buffer) >= buffer_size_limit:
                combined_df = pd.concat(write_buffer, ignore_index=True)
                combined_df.to_csv(
                    temp_output_file,
                    mode='a',
                    index=False,
                    header=first_chunk
                )
                first_chunk = False
                write_buffer = []
                del combined_df
                gc.collect()

            del era5_chunk, chunk_result
            gc.collect()

        # 写入剩余buffer
        if write_buffer:
            combined_df = pd.concat(write_buffer, ignore_index=True)
            combined_df.to_csv(
                temp_output_file,
                mode='a',
                index=False,
                header=first_chunk
            )
            del combined_df
            gc.collect()

        elapsed_time = (datetime.now() - process_start_time).total_seconds()

        # 🔥 修复：最终状态完整赋值
        progress_dict[worker_id] = {
            'year': year,
            'quarter': quarter,
            'status': '完成',
            'chunk_count': chunk_count,
            'matched_typhoons': total_matched,
            'generated_grids': total_grids,
            'start_time': process_start_time.isoformat(),
            'last_update': datetime.now().isoformat()
        }

        return {
            'year': year,
            'quarter': quarter,
            'success': True,
            'matched_typhoons': total_matched,
            'generated_grids': total_grids,
            'processed_chunks': chunk_count,
            'temp_file': temp_output_file if os.path.exists(temp_output_file) else None,
            'elapsed_time': elapsed_time,
            'error': None
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"

        # 🔥 修复：错误状态完整赋值
        progress_dict[worker_id] = {
            'year': year,
            'quarter': quarter,
            'status': f'失败: {str(e)[:50]}',
            'chunk_count': chunk_count if 'chunk_count' in locals() else 0,
            'matched_typhoons': total_matched if 'total_matched' in locals() else 0,
            'generated_grids': total_grids if 'total_grids' in locals() else 0,
            'start_time': process_start_time.isoformat(),
            'last_update': datetime.now().isoformat()
        }

        return {
            'year': year,
            'quarter': quarter,
            'success': False,
            'matched_typhoons': 0,
            'generated_grids': 0,
            'processed_chunks': 0,
            'temp_file': None,
            'elapsed_time': (datetime.now() - process_start_time).total_seconds(),
            'error': error_msg
        }


def progress_monitor_thread(progress_dict, total_files, start_time, stop_event):
    """进度监控线程"""
    while not stop_event.is_set():
        time.sleep(3)

        current_time = datetime.now()
        elapsed = current_time - start_time

        total_chunks = 0
        total_matched = 0
        total_grids = 0
        completed_files = 0
        active_workers = []

        # 🔥 修复：正确读取共享字典
        for worker_id in list(progress_dict.keys()):
            try:
                info = dict(progress_dict[worker_id])  # 转换为普通字典

                if info['status'] in ['完成', '完成(无数据)']:
                    completed_files += 1
                elif '处理中' in info['status']:
                    active_workers.append(info)

                total_chunks += info.get('chunk_count', 0)
                total_matched += info.get('matched_typhoons', 0)
                total_grids += info.get('generated_grids', 0)
            except:
                continue

        if elapsed.total_seconds() > 0:
            chunk_rate = total_chunks / elapsed.total_seconds()
            grid_rate = total_grids / elapsed.total_seconds()
        else:
            chunk_rate = grid_rate = 0

        print(f"\n⏱️  {str(elapsed).split('.')[0]} | 完成: {completed_files}/{total_files} | "
              f"Chunks: {total_chunks} | 匹配: {total_matched} | 网格: {total_grids:,} | "
              f"速度: {chunk_rate:.2f} c/s")

        if active_workers:
            print(f"🔄 活跃进程 ({len(active_workers)}个):")
            for info in active_workers[:5]:
                print(f"   [{info['year']}Q{info['quarter']}] Chunk: {info['chunk_count']}, "
                      f"匹配: {info['matched_typhoons']}, 网格: {info['generated_grids']:,}")


def multiprocess_era5_processing_quarterly(quarterly_files, ibtracs_full, output_file,
                                           chunk_size=1200000, num_workers=None):
    """多进程处理ERA5季度文件"""
    print("\n" + "=" * 80)
    print("🚀 多进程处理ERA5季度文件（50×50标准方案）")
    print("=" * 80)

    # 确定进程数
    if num_workers is None:
        cpu_cores = cpu_count()
        mem_info = get_system_memory_info()

        max_workers_by_memory = int(mem_info['available'] / 2)
        max_workers_by_cpu = max(1, cpu_cores - 1)

        num_workers = min(max_workers_by_cpu, max_workers_by_memory, len(quarterly_files))

        print(f"\n🔧 自动配置进程数:")
        print(f"   CPU核心数: {cpu_cores}")
        print(f"   可用内存: {mem_info['available']:.1f} GB")
        print(f"   实际使用进程数: {num_workers}")
    else:
        num_workers = min(num_workers, len(quarterly_files))
        print(f"\n🔧 使用指定进程数: {num_workers}")

    # 创建临时目录
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='era5_multiprocess_')
    print(f"\n📁 临时目录: {temp_dir}")

    # 创建Manager和共享字典
    manager = Manager()
    progress_dict = manager.dict()

    # 为每个季度文件准备任务
    tasks = []
    for idx, file_info in enumerate(quarterly_files):
        year = file_info['year']
        quarter = file_info['quarter']

        # 提取该季度的IBTrACS数据
        ibtracs_quarter, _, _ = get_ibtracs_for_quarter(
            ibtracs_full, year, quarter, time_margin_days=5
        )

        # 转换为字典格式
        ibtracs_quarter_dict = ibtracs_quarter.to_dict('list')

        tasks.append((
            file_info,
            ibtracs_quarter_dict,
            temp_dir,
            chunk_size,
            progress_dict,
            idx
        ))

    print(f"\n📊 处理任务:")
    print(f"   总文件数: {len(quarterly_files)}")
    print(f"   并行进程数: {num_workers}")
    print(f"\n{'=' * 80}")
    print("开始处理，实时进度如下：")
    print(f"{'=' * 80}\n")

    start_time = datetime.now()
    results = []

    # 启动进度监控线程
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=progress_monitor_thread,
        args=(progress_dict, len(quarterly_files), start_time, stop_event)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # 使用进程池处理
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(process_single_quarterly_file, tasks):
                results.append(result)

                if result['success']:
                    print(f"\n✅ 完成: {result['year']}Q{result['quarter']} - "
                          f"{result['matched_typhoons']} 匹配, {result['generated_grids']:,} 网格, "
                          f"耗时: {result['elapsed_time']:.1f}秒")
                else:
                    print(f"\n❌ 失败: {result['year']}Q{result['quarter']}")

        # 停止监控线程
        stop_event.set()
        monitor_thread.join(timeout=1)

        processing_time = datetime.now() - start_time

        print(f"\n{'=' * 80}")
        print("📈 处理结果统计")
        print("=" * 80)

        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        print(f"\n✅ 成功: {len(successful)}/{len(results)} 个文件")
        print(f"❌ 失败: {len(failed)}/{len(results)} 个文件")

        if successful:
            total_matched = sum(r['matched_typhoons'] for r in successful)
            total_grids = sum(r['generated_grids'] for r in successful)
            total_chunks = sum(r['processed_chunks'] for r in successful)

            print(f"\n📊 数据统计:")
            print(f"   总匹配台风数: {total_matched:,}")
            print(f"   总生成网格点数: {total_grids:,}")
            print(f"   总处理chunk数: {total_chunks:,}")
            print(f"   总处理时间: {processing_time}")
            print(f"   平均每台风网格点: {total_grids / max(total_matched, 1):.0f}")

        if failed:
            print(f"\n❌ 失败文件详情:")
            for r in failed:
                print(f"   {r['year']}Q{r['quarter']}: {r['error'][:100] if r['error'] else '未知错误'}...")

        # 合并临时文件
        print(f"\n{'=' * 80}")
        print("🔄 合并临时文件")
        print("=" * 80)

        temp_files = [r['temp_file'] for r in successful if r['temp_file'] is not None]

        if temp_files:
            print(f"找到 {len(temp_files)} 个临时文件，开始合并...")

            merge_start = datetime.now()
            first_file = True

            if os.path.exists(output_file):
                os.remove(output_file)

            for idx, temp_file in enumerate(temp_files, 1):
                try:
                    for chunk in pd.read_csv(temp_file, chunksize=100000):
                        chunk.to_csv(
                            output_file,
                            mode='a',
                            index=False,
                            header=first_file
                        )
                        first_file = False

                    os.remove(temp_file)

                    if idx % 10 == 0:
                        print(f"  已合并: {idx}/{len(temp_files)} 个文件")

                except Exception as e:
                    print(f"⚠️ 合并文件失败: {temp_file} - {e}")

            merge_time = datetime.now() - merge_start
            print(f"✅ 合并完成，耗时: {merge_time}")

            try:
                os.rmdir(temp_dir)
                print(f"✅ 已清理临时目录")
            except:
                print(f"⚠️ 临时目录未完全清空: {temp_dir}")

        else:
            print("⚠️ 没有生成任何临时文件")

        return len(successful) > 0

    except Exception as e:
        stop_event.set()
        print(f"❌ 多进程处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_output_file(output_file):
    """验证输出文件"""
    print(f"\n=== 验证输出文件（50×50版本）===")

    if not os.path.exists(output_file):
        print(f"❌ 输出文件不存在: {output_file}")
        return False

    try:
        chunk_size = 1200000
        total_rows = 0
        unique_typhoons = set()
        unique_times = set()

        print("正在验证数据...")

        for chunk in pd.read_csv(output_file, chunksize=chunk_size):
            total_rows += len(chunk)

            if 'typhoon_name' in chunk.columns:
                unique_typhoons.update(chunk['typhoon_name'].dropna().unique())

            if 'time' in chunk.columns:
                unique_times.update(chunk['time'].dropna().unique())

        file_size = os.path.getsize(output_file) / (1024 * 1024)

        print(f"\n✅ 基本信息:")
        print(f"文件大小: {file_size:.2f} MB")
        print(f"总行数: {total_rows:,}")
        print(f"唯一台风数: {len(unique_typhoons):,}")
        print(f"唯一时间点数: {len(unique_times):,}")

        return True

    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main_multiprocess():
    """主函数"""
    print("=" * 80)
    print("台风网格数据处理50×50多进程方案：2500点，0.25°精度）")
    print("=" * 80)

    # 配置参数
    ibtracs_file = "F:/ibtracs_1980Q1_2011Q4.csv"

    era5_data_dirs = [
        "G:/era5_final_output/",
        "G://"
    ]

    start_year = 1980
    start_quarter = 1
    end_year = 2011
    end_quarter = 4

    output_file = f"f:/era5_output/typhoon_grid_50x50_0.25deg_{start_year}Q{start_quarter}-{end_year}Q{end_quarter}_multiprocess.csv"

    chunk_size = 1200000
    num_workers = 4

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    start_time = datetime.now()

    try:
        # 步骤1: 查找季度文件
        print(f"\n步骤1: 查找ERA5季度文件...")
        step_start = datetime.now()

        quarterly_files = find_era5_quarterly_files(
            era5_data_dirs,
            start_year, start_quarter,
            end_year, end_quarter
        )

        if len(quarterly_files) == 0:
            print("❌ 未找到任何季度文件")
            return

        step1_time = datetime.now() - step_start
        print(f"步骤1完成，耗时: {step1_time}")

        # 步骤2: 加载完整IBTrACS数据
        print(f"\n步骤2: 加载完整IBTrACS数据...")
        step_start = datetime.now()

        ibtracs_full = load_ibtracs_data(ibtracs_file)

        if len(ibtracs_full) == 0:
            print("❌ IBTrACS数据加载失败")
            return

        step2_time = datetime.now() - step_start
        print(f"步骤2完成，耗时: {step2_time}")

        # 步骤3: 多进程处理
        print(f"\n步骤3: 多进程处理ERA5季度文件...")
        step_start = datetime.now()

        success = multiprocess_era5_processing_quarterly(
            quarterly_files, ibtracs_full, output_file, chunk_size, num_workers
        )

        if not success:
            print("❌ 多进程处理失败")
            return

        step3_time = datetime.now() - step_start
        print(f"步骤3完成，耗时: {step3_time}")

        # 步骤4: 验证输出文件
        print(f"\n步骤4: 验证输出文件...")
        step_start = datetime.now()

        verify_output_file(output_file)

        step4_time = datetime.now() - step_start
        print(f"步骤4完成，耗时: {step4_time}")

        # 总结
        total_time = datetime.now() - start_time
        print(f"\n{'=' * 80}")
        print(f"🎉 全部处理完成！")
        print(f"{'=' * 80}")
        print(f"输出文件: {output_file}")
        print(f"总处理时间: {total_time}")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"❌ 主流程发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"流式处理开始: {start_time}")
    print(f"初始内存使用: {get_memory_usage():.1f} MB")

    mem_info = get_system_memory_info()
    print(f"系统内存: {mem_info['total']:.1f} GB 总量, {mem_info['available']:.1f} GB 可用")
    print(f"CPU核心数: {cpu_count()}")

    try:
        main_multiprocess()

    except KeyboardInterrupt:
        print("\n⚠️ 处理被用户中断!")
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        end_time = datetime.now()
        print(f"\n处理结束: {end_time}")
        print(f"总持续时间: {end_time - start_time}")
        print(f"最终内存使用: {get_memory_usage():.1f} MB")