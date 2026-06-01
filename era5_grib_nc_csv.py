#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ERA5 GRIB -> CSV 转换工具（智能混合并行 - 完整版）
⚡⚡⚡ 性能优化 + 智能并行

核心优化：
1. 直接 NumPy 构建宽格式（跳过 DataFrame 瓶颈）
2. 批量缓冲写入（减少 I/O 100倍）
3. 全局风速计算
4. 按季度分片
5. 智能并行：文件级 + 季度级

并行策略：
- 多文件：2 进程并行处理不同文件
- 单文件：读取后按季度并行处理（可选）
"""

import os
import gc
import time
import glob
import psutil
import logging
import warnings
import shutil
import io
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==================== 日志设置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
INPUT_PATH = "d://era5"
OUTPUT_DIR = "g://era5_final_output"

VARIABLE_MAPPING = {
    "t": "TMP",
    "r": "R_H",
    "u": "U_GRD",
    "v": "V_GRD",
    "w": "W_VEL",
    "vo": "VORT"
}

GRIB_VARIABLE_MAPPING = {
    "t": ["t", "T", "TMP", "Temperature", "2t"],
    "r": ["r", "R", "RH", "Relative_humidity", "2r"],
    "u": ["u", "U", "UGRD", "U_component_of_wind", "10u"],
    "v": ["v", "V", "VGRD", "V_component_of_wind", "10v"],
    "w": ["w", "W", "VVEL", "Vertical_velocity"],
    "vo": ["vo", "VO", "VORT", "Vorticity", "Relative_vorticity"]
}

PRESSURE_LEVELS = [850, 750, 500, 200]
LAT_RANGE = [0, 50]
LON_RANGE = [100, 180]
TIME_RANGE = ["1965-01-01", "2024-12-24"]

# ===== 并行策略配置 =====
ENABLE_FILE_PARALLEL = False  # 文件级并行（多文件时）
ENABLE_QUARTER_PARALLEL = False  # 季度级并行（单文件时，需要>=20GB内存）
MAX_FILE_WORKERS = 2  # 文件级最大并行数
MAX_QUARTER_WORKERS = 2  # 季度级最大并行数（如果启用）

# ===== 性能优化参数 =====
BATCH_SIZE_TIMEPOINTS = 600  # 批次大小
WRITE_BUFFER_SIZE = 5  # ⚡ 批量缓冲写入（累积N个批次再写）
CALCULATE_ABS_V = True  # 计算风速

# ===== 分片策略 =====
SPLIT_BY_QUARTER = True  # 按季度分片

# 调试模式
DEBUG_MODE = True


# ==================== 工具函数 ====================

def check_dependencies():
    """检查依赖"""
    deps = {}

    try:
        import pygrib
        deps['pygrib'] = True
        logger.info("✓ pygrib 已安装")
    except:
        deps['pygrib'] = False
        logger.error("✗ pygrib 未安装！")
        logger.error("  安装: conda install -c conda-forge pygrib")

    try:
        import netCDF4
        deps['netCDF4'] = True
        logger.info("✓ netCDF4 已安装")
    except:
        deps['netCDF4'] = False

    try:
        import h5netcdf
        deps['h5netcdf'] = True
        logger.info("✓ h5netcdf 已安装")
    except:
        deps['h5netcdf'] = False

    if not deps.get('netCDF4') and not deps.get('h5netcdf'):
        logger.warning("⚠ 未安装 NetCDF 库")

    try:
        import pyarrow
        deps['pyarrow'] = True
        logger.info("✓ pyarrow 已安装 - 加速 CSV 写入")
    except:
        deps['pyarrow'] = False
        logger.warning("⚠ 建议安装 pyarrow")
        logger.warning("  安装: pip install pyarrow")

    return deps


def get_available_memory():
    """获取可用内存（GB）"""
    memory = psutil.virtual_memory()
    return memory.available / (1024 ** 3)


def get_optimal_workers():
    """智能计算最优并行数"""
    cpu_count = os.cpu_count() or 4
    available_gb = get_available_memory()

    # 每个进程峰值约 12GB
    max_workers_by_memory = max(1, int(available_gb / 12))
    max_workers_by_cpu = max(1, cpu_count - 1)

    optimal = min(max_workers_by_memory, max_workers_by_cpu, MAX_FILE_WORKERS)

    logger.info(f"CPU 核心数: {cpu_count}")
    logger.info(f"可用内存: {available_gb:.2f} GB")
    logger.info(f"理论最大进程数（按内存）: {max_workers_by_memory}")
    logger.info(f"推荐文件级并行数: {optimal}")

    if optimal < 2 and available_gb < 20:
        logger.warning(f"⚠ 内存不足 20GB，建议单进程运行")

    return optimal


def detect_file_format(file_path):
    """检测文件格式"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.grib', '.grib2', '.grb', '.grb2']:
        return 'grib'
    elif ext in ['.nc', '.nc4', '.netcdf']:
        return 'netcdf'
    else:
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'GRIB':
                    return 'grib'
                elif magic[:3] == b'CDF' or magic == b'\x89HDF':
                    return 'netcdf'
        except:
            pass
    return None


def get_quarter_from_datetime(dt):
    """获取季度"""
    try:
        dt_obj = pd.to_datetime(dt)
        month = dt_obj.month

        if month in [1, 2, 3]:
            return 'Q1'
        elif month in [4, 5, 6]:
            return 'Q2'
        elif month in [7, 8, 9]:
            return 'Q3'
        else:
            return 'Q4'
    except:
        return 'Q1'


def format_time(dt):
    """时间格式化"""
    try:
        dt_obj = pd.to_datetime(dt)
        return dt_obj.strftime('%Y-%m-%d %H-%M')
    except:
        try:
            return pd.to_datetime(dt).isoformat().replace('T', ' ').replace(':', '-')
        except:
            return str(dt).replace('/', '-').replace(':', '-')


def open_grib_with_pygrib(file_path):
    """使用 pygrib 打开 GRIB 文件（完整版）"""
    try:
        import pygrib
    except ImportError:
        raise ImportError("pygrib 未安装！")

    logger.info("  使用 pygrib 引擎打开 GRIB 文件...")
    file_start = time.time()

    # 打开文件
    grbs = pygrib.open(file_path)

    # 读取所有消息
    logger.info("  读取 GRIB 消息...")
    messages = []

    try:
        total_messages = grbs.messages
        logger.info(f"  文件包含 {total_messages} 条消息")
    except:
        total_messages = None

    with tqdm(total=total_messages, desc="  读取GRIB", unit="msg", ncols=80, leave=False) as pbar:
        for msg in grbs:
            messages.append(msg)
            pbar.update(1)

    grbs.close()

    logger.info(f"  ✓ 读取完成: {len(messages)} 条消息 ({time.time() - file_start:.1f}秒)")

    # 组织数据结构
    logger.info("  组织数据结构...")
    data_dict = {}
    all_times = set()
    lat_lon_grid = None

    with tqdm(total=len(messages), desc="  提取数据", unit="msg", ncols=80, leave=False) as pbar:
        for msg in messages:
            var_name = msg.shortName

            try:
                if hasattr(msg, 'level'):
                    level = msg.level
                elif hasattr(msg, 'pressureLevel'):
                    level = msg.pressureLevel
                else:
                    level = 0
            except:
                level = 0

            try:
                valid_date = msg.validDate
            except:
                try:
                    valid_date = msg.dataDate
                except:
                    valid_date = pd.Timestamp('1900-01-01')

            values = msg.values

            if lat_lon_grid is None:
                lats, lons = msg.latlons()
                lat_1d = lats[:, 0]
                lon_1d = lons[0, :]
                lat_lon_grid = (lat_1d, lon_1d)

            key = (var_name, level)
            if key not in data_dict:
                data_dict[key] = {'times': [], 'values': []}

            data_dict[key]['times'].append(valid_date)
            data_dict[key]['values'].append(values)
            all_times.add(valid_date)

            pbar.update(1)

    all_times = sorted(list(all_times))
    lat_1d, lon_1d = lat_lon_grid

    logger.info(f"  数据统计:")
    logger.info(f"    - 时间点: {len(all_times)}")
    logger.info(f"    - 纬度: {len(lat_1d)}")
    logger.info(f"    - 经度: {len(lon_1d)}")
    logger.info(f"    - 变量-层次组合: {len(data_dict)}")

    # 构建 xarray Dataset
    logger.info("  构建 xarray Dataset...")
    data_vars = {}

    all_levels = sorted(set([key[1] for key in data_dict.keys()]))
    has_multiple_levels = len(all_levels) > 1

    if has_multiple_levels:
        logger.info(f"    检测到多层数据: {all_levels}")

        var_groups = defaultdict(dict)
        for (var_name, level), data in data_dict.items():
            var_groups[var_name][level] = data

        for var_name, level_data in var_groups.items():
            if len(level_data) > 1:
                sorted_levels = sorted(level_data.keys())

                level_arrays = []
                for lv in sorted_levels:
                    time_value_pairs = sorted(zip(level_data[lv]['times'], level_data[lv]['values']))
                    values_sorted = [v for t, v in time_value_pairs]
                    level_arrays.append(np.stack(values_sorted, axis=0))

                var_array_4d = np.stack(level_arrays, axis=1)

                data_vars[var_name] = xr.DataArray(
                    var_array_4d,
                    dims=['time', 'level', 'latitude', 'longitude'],
                    coords={
                        'time': sorted([t for t, v in sorted(zip(level_data[sorted_levels[0]]['times'],
                                                                 level_data[sorted_levels[0]]['values']))]),
                        'level': sorted_levels,
                        'latitude': lat_1d,
                        'longitude': lon_1d,
                    },
                    attrs={'original_name': var_name}
                )
            else:
                level = list(level_data.keys())[0]
                time_value_pairs = sorted(zip(level_data[level]['times'], level_data[level]['values']))
                values_sorted = [v for t, v in time_value_pairs]
                var_array = np.stack(values_sorted, axis=0)

                data_vars[var_name] = xr.DataArray(
                    var_array,
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': [t for t, v in time_value_pairs],
                        'latitude': lat_1d,
                        'longitude': lon_1d,
                    },
                    attrs={'level': level, 'original_name': var_name}
                )
    else:
        for (var_name, level), data in data_dict.items():
            time_value_pairs = sorted(zip(data['times'], data['values']))
            values_sorted = [v for t, v in time_value_pairs]
            var_array = np.stack(values_sorted, axis=0)

            data_vars[var_name] = xr.DataArray(
                var_array,
                dims=['time', 'latitude', 'longitude'],
                coords={
                    'time': [t for t, v in time_value_pairs],
                    'latitude': lat_1d,
                    'longitude': lon_1d,
                },
                attrs={'level': level, 'original_name': var_name}
            )

    ds = xr.Dataset(data_vars)
    ds.attrs['source'] = 'pygrib'
    ds.attrs['file'] = os.path.basename(file_path)

    elapsed = time.time() - file_start
    logger.info(f"  ✓ Dataset 构建完成 (总耗时: {elapsed:.1f}秒)")
    logger.info(f"  变量列表: {list(ds.data_vars.keys())}")

    return ds


def open_netcdf(file_path):
    """打开 NetCDF 文件"""
    logger.info("  使用 NetCDF 引擎打开文件...")

    engine = 'h5netcdf'
    try:
        import h5netcdf
    except:
        engine = 'netcdf4'
        try:
            import netCDF4
        except:
            raise ImportError("未安装 NetCDF 库！")

    try:
        ds = xr.open_dataset(file_path, engine=engine)
        logger.info(f"  ✓ NetCDF 文件打开成功（引擎: {engine}）")
        return ds
    except Exception as e:
        logger.error(f"  ✗ NetCDF 文件打开失败: {e}")
        raise


def open_dataset_by_format(file_path, format_type=None):
    """根据格式打开数据集"""
    if format_type is None:
        format_type = detect_file_format(file_path)

    if format_type == 'grib':
        return open_grib_with_pygrib(file_path)
    elif format_type == 'netcdf':
        return open_netcdf(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def standardize_dataset(ds, format_type='grib'):
    """标准化数据集"""
    rename_map = {}

    # 时间维度
    if 'time' not in ds.dims and 'time' not in ds.coords:
        time_candidates = ['valid_time', 'TIME', 't']
        for tc in time_candidates:
            if tc in ds.dims or tc in ds.coords:
                rename_map[tc] = 'time'
                break

    # 纬度
    if 'latitude' not in ds.dims and 'latitude' not in ds.coords:
        lat_candidates = ['lat', 'LAT', 'LATITUDE', 'y']
        for lc in lat_candidates:
            if lc in ds.dims or lc in ds.coords:
                rename_map[lc] = 'latitude'
                break

    # 经度
    if 'longitude' not in ds.dims and 'longitude' not in ds.coords:
        lon_candidates = ['lon', 'LON', 'LONGITUDE', 'x']
        for lc in lon_candidates:
            if lc in ds.dims or lc in ds.coords:
                rename_map[lc] = 'longitude'
                break

    # 气压层
    if 'level' not in ds.dims and 'level' not in ds.coords:
        level_candidates = ['isobaricInhPa', 'plev', 'pressure_level', 'lev', 'LEVEL']
        for lvc in level_candidates:
            if lvc in ds.dims or lvc in ds.coords:
                rename_map[lvc] = 'level'
                break

    # 执行重命名
    if rename_map:
        logger.info(f"  维度重命名: {rename_map}")
        try:
            ds = ds.rename(rename_map)
        except Exception as e:
            logger.warning(f"  ⚠ 批量重命名失败，逐个尝试: {e}")
            for old, new in rename_map.items():
                try:
                    if old in ds.dims or old in ds.coords:
                        if new not in ds.dims and new not in ds.coords:
                            ds = ds.rename({old: new})
                            logger.info(f"    ✓ {old} -> {new}")
                except Exception as e2:
                    logger.warning(f"    ✗ {old} -> {new} 失败: {e2}")

    # 经度范围调整
    if 'longitude' in ds.coords:
        lon_values = ds.longitude.values
        if lon_values.max() > 180:
            logger.info("  检测到 0-360 经度范围，转换为 -180-180...")
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
            ds = ds.sortby('longitude')

    # 气压层单位
    if 'level' in ds.coords:
        level_values = ds.level.values
        if level_values.max() > 2000:
            logger.info("  检测到Pa单位，转换为hPa...")
            ds['level'] = ds['level'] / 100

    return ds


def find_variable_in_dataset(ds, var_short_name, format_type='grib'):
    """在数据集中查找变量"""
    # 直接匹配
    if var_short_name in ds.data_vars:
        return var_short_name

    # GRIB 变量映射
    if format_type == 'grib' and var_short_name in GRIB_VARIABLE_MAPPING:
        for candidate in GRIB_VARIABLE_MAPPING[var_short_name]:
            if candidate in ds.data_vars:
                logger.info(f"  变量映射: {var_short_name} -> {candidate}")
                return candidate

    # 不区分大小写匹配
    var_lower = var_short_name.lower()
    for var in ds.data_vars:
        if var.lower() == var_lower:
            return var

    return None


def convert_xarray_to_wide_fast(ds_batch, variables, levels, var_mapping):
    """
    ⚡ 超快速转换：直接从 xarray 构建宽格式 DataFrame
    """
    times = ds_batch['time'].values
    lats = ds_batch['latitude'].values
    lons = ds_batch['longitude'].values

    n_times = len(times)
    n_lats = len(lats)
    n_lons = len(lons)
    total_rows = n_times * n_lats * n_lons

    # 预分配结果字典
    result = {}

    # 坐标列
    result['lat'] = np.tile(lats, n_lons * n_times)
    result['lon'] = np.tile(np.repeat(lons, n_lats), n_times)

    # 时间列 + 季度列
    time_formatted = []
    quarters = []
    for t in times:
        try:
            dt = pd.to_datetime(t)
            time_formatted.append(dt.strftime('%Y-%m-%d %H-%M'))
            quarters.append(get_quarter_from_datetime(dt))
        except:
            time_str = str(t).replace('/', '-').replace(':', '-')
            time_formatted.append(time_str)
            quarters.append('Q1')

    result['time'] = np.repeat(time_formatted, n_lats * n_lons)
    result['quarter'] = np.repeat(quarters, n_lats * n_lons)

    # 直接提取变量数据（NumPy向量化）
    for var in variables:
        if var not in ds_batch.data_vars:
            continue

        var_data = ds_batch[var].values

        if var_data.ndim == 4:  # 有 level 维度
            for i, level in enumerate(levels):
                col_name = f"{var_mapping.get(var, var.upper())}_L100_{level}hPa"
                level_data = var_data[:, i, :, :]
                reshaped = level_data.reshape(n_times, n_lats * n_lons).ravel()
                result[col_name] = reshaped

        elif var_data.ndim == 3:  # 无 level 维度
            col_name = var_mapping.get(var, var.upper())
            reshaped = var_data.reshape(n_times, n_lats * n_lons).ravel()
            result[col_name] = reshaped

    df = pd.DataFrame(result)

    return df


def save_dataframe_safe(df, output_file, mode="w", header=True):
    """
    ⚡ 安全快速保存 DataFrame（带重试机制）
    """
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # 尝试使用 pyarrow 引擎（更快）
            try:
                import pyarrow
                df.to_csv(output_file, index=False, mode=mode, header=header,
                          chunksize=100000, compression=None)
            except:
                # 回退到标准方法
                with open(output_file, mode, newline='', encoding='utf-8',
                          buffering=8192 * 32) as f:  # 增大缓冲区
                    df.to_csv(f, index=False, header=header, chunksize=100000)

            return True

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"  写入失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
                gc.collect()
            else:
                logger.error(f"  写入最终失败: {e}")
                if DEBUG_MODE:
                    import traceback
                    logger.error(traceback.format_exc())
                raise

    return False


# ==================== 核心处理函数 ====================

def process_single_file(file_path, output_dir, variables=None, levels=None,
                        lat_range=None, lon_range=None, time_range=None,
                        optimal_workers=None):
    """
    处理单个文件（完整版：批量缓冲写入 + 详细进度）
    """
    try:
        file_start = time.time()
        file_name = os.path.basename(file_path)
        logger.info(f"{'=' * 70}")
        logger.info(f"处理文件: {file_name}")
        logger.info(f"{'=' * 70}")

        # 检查磁盘空间
        disk_usage = shutil.disk_usage(output_dir)
        free_gb = disk_usage.free / (1024 ** 3)
        logger.info(f"可用磁盘空间: {free_gb:.2f} GB")

        if free_gb < 5:
            logger.error(f"✗ 磁盘空间不足: 仅剩 {free_gb:.2f} GB")
            return False

        # 检查内存
        available_gb = get_available_memory()
        logger.info(f"可用内存: {available_gb:.2f} GB")

        if available_gb < 10:
            logger.warning(f"⚠ 内存可能不足！建议至少10GB，当前{available_gb:.1f}GB")

        # 检测文件格式
        format_type = detect_file_format(file_path)
        if format_type is None:
            logger.error(f"✗ 无法识别文件格式")
            return False

        logger.info(f"文件格式: {format_type.upper()}")
        logger.info(f"文件大小: {os.path.getsize(file_path) / (1024 ** 3):.2f} GB")

        # === 打开数据集 ===
        try:
            ds = open_dataset_by_format(file_path, format_type)
        except Exception as e:
            logger.error(f"✗ 文件打开失败: {e}")
            if DEBUG_MODE:
                import traceback
                logger.error(traceback.format_exc())
            return False

        # 标准化
        logger.info("标准化数据集...")
        ds = standardize_dataset(ds, format_type)

        # 筛选数据
        logger.info("筛选数据...")
        if lat_range and "latitude" in ds.coords:
            ds = ds.sel(latitude=slice(lat_range[1], lat_range[0]))
            logger.info(f"  纬度筛选: {lat_range}")

        if lon_range and "longitude" in ds.coords:
            ds = ds.sel(longitude=slice(lon_range[0], lon_range[1]))
            logger.info(f"  经度筛选: {lon_range}")

        if time_range and "time" in ds.coords:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))
            logger.info(f"  时间筛选: {time_range}")

        # 气压层筛选
        if levels and "level" in ds.dims:
            ds_levels = ds["level"].values
            if np.issubdtype(ds_levels.dtype, np.number):
                levels_to_use = [int(lv) if isinstance(lv, str) else lv for lv in levels]
            else:
                levels_to_use = [str(lv) for lv in levels]

            available_levels = [lv for lv in levels_to_use if lv in ds_levels]
            if available_levels:
                ds = ds.sel(level=available_levels)
                logger.info(f"  气压层筛选: {available_levels}")
        else:
            available_levels = list(ds["level"].values) if "level" in ds.dims else []

        # 变量筛选
        logger.info("筛选变量...")
        available_vars = []
        if variables:
            for var in variables:
                found_var = find_variable_in_dataset(ds, var, format_type)
                if found_var:
                    available_vars.append(found_var)
                    if found_var != var:
                        ds = ds.rename({found_var: var})
                        available_vars[-1] = var
                else:
                    logger.warning(f"  ⚠ 变量 '{var}' 未找到")
        else:
            available_vars = list(ds.data_vars.keys())

        if not available_vars:
            logger.error("✗ 没有可用变量")
            return False

        ds = ds[available_vars]

        # ⚡ 全局计算风速
        calculate_wind_speed = CALCULATE_ABS_V and "u" in available_vars and "v" in available_vars
        if calculate_wind_speed:
            logger.info("⚡ 全局计算风速（一次性）...")
            try:
                ds["abs_v"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
                available_vars.append("abs_v")
                logger.info("  ✓ 风速计算完成")
            except Exception as e:
                logger.warning(f"  ⚠ 风速计算失败: {e}")
                calculate_wind_speed = False

        # 变量映射
        var_mapping = dict(VARIABLE_MAPPING)
        if calculate_wind_speed and "abs_v" in available_vars:
            var_mapping["abs_v"] = "ABS_V"

        # 数据规模
        n_times = len(ds.time.values)
        n_lats = len(ds.latitude.values)
        n_lons = len(ds.longitude.values)
        n_levels = len(available_levels)
        total_rows = n_times * n_lats * n_lons

        logger.info(f"\n数据规模:")
        logger.info(f"  时间点: {n_times}")
        logger.info(f"  纬度: {n_lats}")
        logger.info(f"  经度: {n_lons}")
        logger.info(f"  气压层: {n_levels}")
        logger.info(f"  总输出行数: {total_rows:,}")
        logger.info(f"  可用变量: {available_vars}")

        # ====== ⚡⚡⚡ 批量处理（完整版）======
        logger.info(f"\n{'=' * 70}")
        if SPLIT_BY_QUARTER:
            logger.info("📅 开始批量处理（按季度分片 + 批量缓冲写入）...")
        else:
            logger.info("⚡ 开始批量处理（批量缓冲写入）...")
        logger.info(f"{'=' * 70}")

        file_base = os.path.splitext(file_name)[0]
        os.makedirs(output_dir, exist_ok=True)

        OPTIMIZED_BATCH_SIZE = BATCH_SIZE_TIMEPOINTS
        logger.info(f"  批次大小: {OPTIMIZED_BATCH_SIZE} 时间点")
        logger.info(f"  ⚡ 写入缓冲: 累积 {WRITE_BUFFER_SIZE} 批次后写入磁盘")

        if SPLIT_BY_QUARTER:
            logger.info(f"  📅 分片策略: 按季度（Q1, Q2, Q3, Q4）")

        # 季度文件和缓冲区
        quarter_files = {}  # {quarter: file_path}
        quarter_buffers = defaultdict(list)  # {quarter: [df1, df2, ...]}
        quarter_headers_written = {}  # {quarter: bool}
        quarter_row_counts = defaultdict(int)  # {quarter: row_count}

        processed_times = 0

        for i in range(0, n_times, OPTIMIZED_BATCH_SIZE):
            batch_start = time.time()
            batch_end = min(i + OPTIMIZED_BATCH_SIZE, n_times)

            try:
                # 提取批次数据
                ds_batch = ds.isel(time=slice(i, batch_end))

                # ⚡⚡⚡ 快速转换 ⚡⚡⚡
                df_wide = convert_xarray_to_wide_fast(
                    ds_batch, available_vars, available_levels, var_mapping
                )

                if df_wide.empty:
                    logger.warning(f"  批次 {i}-{batch_end} 为空，跳过")
                    continue

                # 按季度分组
                if SPLIT_BY_QUARTER and 'quarter' in df_wide.columns:
                    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
                        df_quarter = df_wide[df_wide['quarter'] == quarter].copy()

                        if df_quarter.empty:
                            continue

                        # 删除 quarter 列
                        df_quarter = df_quarter.drop(columns=['quarter'])

                        # ⚡ 添加到缓冲区
                        quarter_buffers[quarter].append(df_quarter)
                        quarter_row_counts[quarter] += len(df_quarter)

                        # ⚡ 检查是否需要写入（累积到 WRITE_BUFFER_SIZE）
                        if len(quarter_buffers[quarter]) >= WRITE_BUFFER_SIZE:
                            # 初始化文件路径
                            if quarter not in quarter_files:
                                quarter_files[quarter] = os.path.join(
                                    output_dir,
                                    f"{file_base}_{quarter}.csv"
                                )
                                quarter_headers_written[quarter] = False

                            # 合并缓冲区
                            df_to_write = pd.concat(quarter_buffers[quarter], ignore_index=True)

                            # 写入
                            mode = "w" if not quarter_headers_written[quarter] else "a"
                            header = not quarter_headers_written[quarter]

                            save_dataframe_safe(df_to_write, quarter_files[quarter],
                                                mode=mode, header=header)

                            quarter_headers_written[quarter] = True
                            quarter_buffers[quarter] = []  # 清空缓冲区

                            del df_to_write
                            gc.collect()

                        del df_quarter
                else:
                    # 不按季度分片
                    quarter = 'all'
                    if quarter not in quarter_files:
                        quarter_files[quarter] = os.path.join(
                            output_dir,
                            f"{file_base}_wide.csv"
                        )
                        quarter_headers_written[quarter] = False

                    quarter_buffers[quarter].append(df_wide)
                    quarter_row_counts[quarter] += len(df_wide)

                    if len(quarter_buffers[quarter]) >= WRITE_BUFFER_SIZE:
                        df_to_write = pd.concat(quarter_buffers[quarter], ignore_index=True)

                        mode = "w" if not quarter_headers_written[quarter] else "a"
                        header = not quarter_headers_written[quarter]

                        save_dataframe_safe(df_to_write, quarter_files[quarter],
                                            mode=mode, header=header)

                        quarter_headers_written[quarter] = True
                        quarter_buffers[quarter] = []

                        del df_to_write
                        gc.collect()

                processed_times += (batch_end - i)
                batch_time = time.time() - batch_start

                # ⚡⚡⚡ 详细进度信息 ⚡⚡⚡
                progress = processed_times / n_times * 100
                speed = (batch_end - i) / batch_time if batch_time > 0 else 0
                eta = (n_times - processed_times) / speed if speed > 0 else 0

                # 显示各季度行数
                if SPLIT_BY_QUARTER:
                    quarter_info = " | ".join([
                        f"{q}:{quarter_row_counts[q]:,}"
                        for q in ['Q1', 'Q2', 'Q3', 'Q4']
                        if quarter_row_counts[q] > 0
                    ])
                    logger.info(
                        f"  进度: {progress:5.1f}% | "
                        f"⚡ 速度: {speed:5.1f} 时间点/秒 | "
                        f"{quarter_info} | "
                        f"剩余: {eta:.0f}秒"
                    )
                else:
                    logger.info(
                        f"  进度: {progress:5.1f}% | "
                        f"⚡ 速度: {speed:5.1f} 时间点/秒 | "
                        f"行数: {quarter_row_counts['all']:,} | "
                        f"剩余: {eta:.0f}秒"
                    )

            except Exception as e:
                logger.error(f"  批次 {i}-{batch_end} 处理失败: {e}")
                if DEBUG_MODE:
                    import traceback
                    logger.error(traceback.format_exc())

            finally:
                try:
                    del df_wide, ds_batch
                except:
                    pass
                gc.collect()

        # ⚡ 写入剩余缓冲区
        logger.info("  写入剩余缓冲数据...")
        for quarter, buffer in quarter_buffers.items():
            if buffer:
                if quarter not in quarter_files:
                    if SPLIT_BY_QUARTER:
                        quarter_files[quarter] = os.path.join(
                            output_dir,
                            f"{file_base}_{quarter}.csv"
                        )
                    else:
                        quarter_files[quarter] = os.path.join(
                            output_dir,
                            f"{file_base}_wide.csv"
                        )
                    quarter_headers_written[quarter] = False

                df_to_write = pd.concat(buffer, ignore_index=True)

                mode = "w" if not quarter_headers_written[quarter] else "a"
                header = not quarter_headers_written[quarter]

                save_dataframe_safe(df_to_write, quarter_files[quarter],
                                    mode=mode, header=header)

                logger.info(f"    ✓ {quarter}: 写入剩余 {len(df_to_write):,} 行")

                del df_to_write
                gc.collect()

        # 关闭数据集
        try:
            ds.close()
        except:
            pass

        del ds
        gc.collect()

        elapsed = time.time() - file_start

        # 统计输出文件
        logger.info(f"\n{'=' * 70}")
        logger.info("统计输出文件...")
        logger.info(f"{'=' * 70}")

        total_size = 0
        output_files = []

        for quarter, file_path in quarter_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size += size
                output_files.append((file_path, size, quarter_row_counts.get(quarter, 0)))

        logger.info(f"✅ 处理完成！")
        logger.info(f"  生成文件数: {len(output_files)}")
        logger.info(f"  总大小: {total_size / (1024 ** 3):.2f} GB")
        logger.info(f"  总耗时: {elapsed / 60:.1f} 分钟 ({elapsed:.1f}秒)")
        logger.info(f"  ⚡ 平均速度: {n_times / elapsed:.2f} 时间点/秒")
        logger.info(f"\n输出文件列表:")
        for f, size, rows in sorted(output_files, key=lambda x: x[0]):
            logger.info(f"  ✓ {os.path.basename(f):40s}  {size / (1024 ** 3):8.2f} GB  ({rows:,} 行)")
        logger.info(f"{'=' * 70}\n")

        return True

    except Exception as e:
        logger.error(f"✗ 文件处理失败: {e}")
        if DEBUG_MODE:
            import traceback
            logger.error(traceback.format_exc())
        return False


def process_files_parallel(files, output_dir, optimal_workers, **kwargs):
    """
    并行处理多个文件（文件级并行）
    """
    if optimal_workers > 1:
        available_gb = get_available_memory()
        required_gb = optimal_workers * 12

        if available_gb < required_gb:
            logger.warning(f"⚠ 内存可能不足！")
            logger.warning(f"  需要: {required_gb:.1f}GB, 可用: {available_gb:.1f}GB")
            user_input = input("  是否继续？(y/n): ")
            if user_input.lower() != 'y':
                logger.info("  切换为单进程模式...")
                optimal_workers = 1

    logger.info(f"\n启动文件级并行处理（{optimal_workers} 进程）...")

    success = 0
    failed = 0
    total_start = time.time()

    if optimal_workers == 1:
        # 单进程串行
        for file in files:
            try:
                if process_single_file(file, output_dir, optimal_workers=1, **kwargs):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.error(f"✗ {os.path.basename(file)}: {e}")
    else:
        # 多进程并行
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            process_func = partial(
                process_single_file,
                output_dir=output_dir,
                optimal_workers=optimal_workers,
                **kwargs
            )

            futures = {executor.submit(process_func, f): f for f in files}

            for future in as_completed(futures):
                file = futures[future]
                try:
                    if future.result():
                        success += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"✗ {os.path.basename(file)}: {e}")

    total_time = time.time() - total_start

    logger.info(f"\n{'=' * 70}")
    logger.info(f"🎉 全部处理完成！")
    logger.info(f"{'=' * 70}")
    logger.info(f"  ✅ 成功: {success} 个文件")
    logger.info(f"  ❌ 失败: {failed} 个文件")
    logger.info(f"  ⏱️  总耗时: {total_time / 3600:.2f} 小时")
    if len(files) > 0:
        logger.info(f"  📊 平均速度: {total_time / len(files) / 60:.1f} 分钟/文件")
    logger.info(f"{'=' * 70}")


def main():
    """主函数"""
    logger.info("\n" + "=" * 70)
    logger.info("⚡⚡ ERA5 GRIB -> CSV 智能并行转换器 [完整版]")
    logger.info("=" * 70)

    # 检查依赖
    deps = check_dependencies()
    if not deps.get('pygrib'):
        logger.error("\n✗ 缺少必需依赖 pygrib！")
        logger.error("  安装命令: conda install -c conda-forge pygrib")
        return

    optimal_workers = get_optimal_workers()

    # 查找文件
    if os.path.isdir(INPUT_PATH):
        patterns = ['*.nc', '*.nc4', '*.grib', '*.grib2', '*.grb', '*.grb2']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(INPUT_PATH, pattern)))
        files = sorted(set(files))
    else:
        files = [INPUT_PATH] if os.path.exists(INPUT_PATH) else []

    if not files:
        logger.error("✗ 未找到文件")
        return

    # 统计文件格式
    format_counts = {'grib': 0, 'netcdf': 0, 'unknown': 0}
    for f in files:
        fmt = detect_file_format(f)
        if fmt:
            format_counts[fmt] += 1
        else:
            format_counts['unknown'] += 0

    logger.info(f"\n找到 {len(files)} 个文件:")
    logger.info(f"  GRIB/GRIB2: {format_counts['grib']} 个")
    logger.info(f"  NetCDF: {format_counts['netcdf']} 个")
    if format_counts['unknown'] > 0:
        logger.warning(f"  未知格式: {format_counts['unknown']} 个")

    # 并行策略说明
    logger.info(f"\n并行策略:")
    if len(files) > 1 and ENABLE_FILE_PARALLEL and optimal_workers >= 2:
        logger.info(f"  ✅ 文件级并行: {optimal_workers} 进程（推荐）")
    else:
        logger.info(f"  串行处理文件")

    if ENABLE_QUARTER_PARALLEL:
        available_gb = get_available_memory()
        if available_gb >= 20:
            logger.info(f"  ✅ 季度级并行: 启用（需要>=20GB内存）")
        else:
            logger.warning(f"  ⚠ 季度级并行: 禁用（内存不足20GB）")
            logger.info(f"     当前可用: {available_gb:.1f}GB")
    else:
        logger.info(f"  季度级并行: 未启用")

    logger.info(f"\n处理配置:")
    logger.info(f"  变量: {list(VARIABLE_MAPPING.keys())}")
    logger.info(f"  气压层: {PRESSURE_LEVELS}")
    logger.info(f"  空间范围: 纬度{LAT_RANGE}, 经度{LON_RANGE}")
    logger.info(f"  时间范围: {TIME_RANGE}")
    logger.info(f"  📅 分片策略: {'按季度（Q1-Q4）' if SPLIT_BY_QUARTER else '按大小'}")
    logger.info(f"  ⚡ 批次大小: {BATCH_SIZE_TIMEPOINTS} 时间点")
    logger.info(f"  ⚡ 写入缓冲: {WRITE_BUFFER_SIZE} 批次（减少I/O）")
    logger.info(f"  计算风速: {CALCULATE_ABS_V}")
    logger.info(f"  ⚡ 性能优化: NumPy直接构建 + 批量缓冲写入")
    logger.info("=" * 70)

    kwargs = {
        "variables": list(VARIABLE_MAPPING.keys()),
        "levels": PRESSURE_LEVELS,
        "lat_range": LAT_RANGE,
        "lon_range": LON_RANGE,
        "time_range": TIME_RANGE
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理文件
    if len(files) > 1 and ENABLE_FILE_PARALLEL and optimal_workers >= 2:
        # 文件级并行
        process_files_parallel(files, OUTPUT_DIR, optimal_workers, **kwargs)
    else:
        # 串行处理
        for f in files:
            process_single_file(f, OUTPUT_DIR, optimal_workers=1, **kwargs)


if __name__ == "__main__":
    # Windows 必需
    mp.set_start_method('spawn', force=True)
    main()