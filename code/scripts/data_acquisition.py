# File: ~/meteorology_analysis/code/scripts/data_acquisition.py
"""
气象数据获取工作流程
支持ERA5, NOAA, CMA等数据源
"""
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MeteorologicalDataAcquirer:
    """气象数据获取器"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def create_sample_data(self, variable='temperature', 
                          start_date='2024-01-01', 
                          end_date='2024-01-31'):
        """
        创建模拟气象数据（在没有真实数据时使用）
        
        Parameters:
        -----------
        variable : str
            变量名: 'temperature', 'pressure', 'humidity', 'wind'
        start_date : str
            开始日期
        end_date : str
            结束日期
        """
        print(f"创建模拟{variable}数据: {start_date} 到 {end_date}")
        
        # 创建时间和空间网格
        times = pd.date_range(start_date, end_date, freq='D')
        lats = np.linspace(-90, 90, 181)  # 2度分辨率
        lons = np.linspace(-180, 180, 361)  # 1度分辨率
        
        # 根据变量创建不同的数据
        if variable == 'temperature':
            # 温度：纬度效应 + 季节变化 + 日变化 + 随机扰动
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            base_temp = 15 * np.cos(np.deg2rad(lat_grid))
            
            data = np.zeros((len(times), len(lats), len(lons)))
            for i, t in enumerate(times):
                # 季节效应（模拟）
                season_effect = 10 * np.sin(2 * np.pi * (t.dayofyear - 80) / 365.25)
                # 日变化效应
                diurnal_effect = 5 * np.sin(2 * np.pi * t.hour / 24) if hasattr(t, 'hour') else 0
                # 随机扰动
                random_effect = np.random.randn(*lat_grid.shape) * 3
                
                data[i, :, :] = base_temp + season_effect + diurnal_effect + random_effect
            
            units = '°C'
            long_name = '2m Temperature'
            
        elif variable == 'pressure':
            # 海平面气压
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
            base_pressure = 1013.25  # hPa
            
            data = np.zeros((len(times), len(lats), len(lons)))
            for i, t in enumerate(times):
                # 纬度效应
                lat_effect = 20 * np.sin(np.deg2rad(lat_grid))
                # 随机扰动
                random_effect = np.random.randn(*lat_grid.shape) * 5
                
                data[i, :, :] = base_pressure + lat_effect + random_effect
            
            units = 'hPa'
            long_name = 'Sea Level Pressure'
            
        else:
            raise ValueError(f"不支持的变量: {variable}")
        
        # 创建xarray Dataset
        ds = xr.Dataset(
            {
                variable: (['time', 'lat', 'lon'], data)
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            },
            attrs={
                'title': f'Simulated {variable} data',
                'source': 'Generated for meteorological analysis practice',
                'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'author': 'Meteorology Student',
                'variable': variable,
                'units': units,
                'long_name': long_name
            }
        )
        
        # 保存为NetCDF
        filename = f"{variable}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"
        filepath = self.data_dir / filename
        ds.to_netcdf(filepath)
        print(f"✅ 数据已保存: {filepath}")
        
        return ds
    
    def load_netcdf_data(self, filepath):
        """加载NetCDF数据"""
        print(f"加载数据: {filepath}")
        ds = xr.open_dataset(filepath)
        
        print("数据基本信息:")
        print(f"  维度: {dict(ds.dims)}")
        print(f"  变量: {list(ds.data_vars.keys())}")
        print(f"  时间范围: {ds.time.min().values} 到 {ds.time.max().values}")
        
        return ds
    
    def preprocess_data(self, ds, variable=None):
        """数据预处理"""
        print("数据预处理...")
        
        # 1. 重命名变量（如果需要）
        if variable and variable in ds.data_vars:
            ds = ds.rename({variable: 'data'})
        
        # 2. 处理缺失值
        if 'data' in ds.data_vars:
            ds['data'] = ds['data'].fillna(0)
        
        # 3. 添加坐标属性
        if 'lat' in ds.coords:
            ds.lat.attrs['units'] = 'degrees_north'
            ds.lat.attrs['long_name'] = 'latitude'
        
        if 'lon' in ds.coords:
            ds.lon.attrs['units'] = 'degrees_east'
            ds.lon.attrs['long_name'] = 'longitude'
        
        # 4. 计算统计量
        if 'data' in ds.data_vars:
            stats = {
                'mean': float(ds.data.mean().values),
                'std': float(ds.data.std().values),
                'min': float(ds.data.min().values),
                'max': float(ds.data.max().values)
            }
            print(f"  统计量: {stats}")
        
        return ds

# 使用示例
if __name__ == "__main__":
    # 初始化数据获取器
    acquirer = MeteorologicalDataAcquirer()
    
    # 创建模拟数据
    temp_data = acquirer.create_sample_data('temperature', '2024-01-01', '2024-01-07')
    pressure_data = acquirer.create_sample_data('pressure', '2024-01-01', '2024-01-07')
    
    # 加载和预处理数据
    loaded_data = acquirer.load_netcdf_data('data/raw/temperature_20240101_20240107.nc')
    processed_data = acquirer.preprocess_data(loaded_data, 'temperature')