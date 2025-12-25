# File: ~/meteorology_analysis/code/scripts/data_visualization.py
"""
气象数据可视化工作流程
包含地图绘制、时间序列图、统计图等
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MeteorologicalVisualizer:
    """气象数据可视化器"""
    
    def __init__(self, output_dir="visualization/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_spatial_distribution(self, ds, variable_name, time_idx=0, 
                                 region=None, save_fig=True):
        """
        绘制空间分布图
        
        Parameters:
        -----------
        ds : xarray.Dataset
            数据集
        variable_name : str
            变量名
        time_idx : int
            时间索引
        region : tuple
            区域限制 (min_lon, max_lon, min_lat, max_lat)
        save_fig : bool
            是否保存图像
        """
        if variable_name not in ds.data_vars:
            print(f"❌ 变量 {variable_name} 不存在")
            return None
        
        # 选择数据
        if 'time' in ds.dims:
            data = ds[variable_name].isel(time=time_idx)
            time_str = f"Time: {ds.time.isel(time=time_idx).values}"
        else:
            data = ds[variable_name]
            time_str = "Time Average"
        
        # 创建图形
        fig = plt.figure(figsize=(12, 8))
        
        # 设置投影
        if region:
            # 区域地图
            proj = ccrs.PlateCarree()
            ax = plt.axes(projection=proj)
            ax.set_extent(region, crs=proj)
        else:
            # 全球地图
            proj = ccrs.PlateCarree(central_longitude=180)
            ax = plt.axes(projection=proj)
            ax.set_global()
        
        # 添加地理特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
        
        # 绘制数据
        if 'lat' in data.dims and 'lon' in data.dims:
            # 二维数据
            im = data.plot(ax=ax, transform=ccrs.PlateCarree(),
                          cmap='RdBu_r', add_colorbar=True,
                          cbar_kwargs={'label': f'{variable_name} ({data.attrs.get("units", "")})',
                                      'orientation': 'horizontal',
                                      'shrink': 0.8})
        else:
            print("❌ 数据不是二维空间数据")
            return None
        
        # 添加标题
        title = f'Spatial Distribution of {variable_name}\n{time_str}'
        plt.title(title, fontsize=14, pad=20)
        
        # 添加网格
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            filename = f"{variable_name}_spatial_distribution.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ 空间分布图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_time_series(self, ds, variable_name, location=None, 
                        save_fig=True, **kwargs):
        """
        绘制时间序列图
        
        Parameters:
        -----------
        ds : xarray.Dataset
            数据集
        variable_name : str
            变量名
        location : tuple
            位置 (lat, lon)，如果为None则用全局平均
        save_fig : bool
            是否保存图像
        """
        if variable_name not in ds.data_vars:
            print(f"❌ 变量 {variable_name} 不存在")
            return None
        
        # 选择数据
        if location:
            lat_idx = np.abs(ds.lat - location[0]).argmin()
            lon_idx = np.abs(ds.lon - location[1]).argmin()
            ts_data = ds[variable_name].isel(lat=lat_idx, lon=lon_idx)
            location_str = f"({location[0]}°N, {location[1]}°E)"
        else:
            ts_data = ds[variable_name].mean(dim=['lat', 'lon'])
            location_str = "Global Average"
        
        # 转换为pandas Series
        ts_series = ts_data.to_pandas()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 时间序列图
        ax1.plot(ts_series.index, ts_series.values, 
                linewidth=2, color='steelblue', label=variable_name)
        
        # 添加移动平均
        if len(ts_series) > 7:
            window_size = min(7, len(ts_series) // 10)
            moving_avg = ts_series.rolling(window=window_size, center=True).mean()
            ax1.plot(moving_avg.index, moving_avg.values, 
                    linewidth=2, color='red', linestyle='--', 
                    label=f'{window_size}-day Moving Average')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel(f'{variable_name} ({ds[variable_name].attrs.get("units", "")})', 
                      fontsize=12)
        ax1.set_title(f'Time Series of {variable_name} at {location_str}', 
                     fontsize=14, pad=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 直方图
        ax2.hist(ts_series.values, bins=30, density=True, 
                alpha=0.6, color='steelblue', edgecolor='black')
        
        # 添加正态分布曲线
        mu, std = ts_series.mean(), ts_series.std()
        x = np.linspace(ts_series.min(), ts_series.max(), 100)
        p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax2.plot(x, p, 'r-', linewidth=2, label=f'Normal Distribution\nμ={mu:.2f}, σ={std:.2f}')
        
        ax2.set_xlabel(f'{variable_name} ({ds[variable_name].attrs.get("units", "")})', 
                      fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'Distribution of {variable_name} at {location_str}', 
                     fontsize=14, pad=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            loc_str = location_str.replace('(', '').replace(')', '').replace('°', '').replace(', ', '_').replace(' ', '_')
            filename = f"{variable_name}_timeseries_{loc_str}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ 时间序列图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_correlation(self, ds, var1, var2, save_fig=True):
        """
        绘制相关性图
        
        Parameters:
        -----------
        ds : xarray.Dataset
            数据集
        var1, var2 : str
            变量名
        save_fig : bool
            是否保存图像
        """
        if var1 not in ds.data_vars or var2 not in ds.data_vars:
            print(f"❌ 变量 {var1} 或 {var2} 不存在")
            return None
        
        # 获取数据
        data1 = ds[var1].values.flatten()
        data2 = ds[var2].values.flatten()
        
        # 计算相关系数
        corr_coef = np.corrcoef(data1, data2)[0, 1]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 散点图
        scatter = ax1.scatter(data1, data2, alpha=0.5, s=10, 
                            c=np.abs(data1), cmap='viridis')
        
        # 添加回归线
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
        x_line = np.array([data1.min(), data1.max()])
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}')
        
        ax1.set_xlabel(f'{var1} ({ds[var1].attrs.get("units", "")})', fontsize=12)
        ax1.set_ylabel(f'{var2} ({ds[var2].attrs.get("units", "")})', fontsize=12)
        ax1.set_title(f'Correlation: {var1} vs {var2}\nr = {corr_coef:.4f}, p = {p_value:.4f}', 
                     fontsize=14, pad=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax1, label=f'{var1} magnitude')
        
        # 2. 热力图（如果数据是2D）
        if len(ds[var1].shape) >= 2:
            # 计算空间相关系数
            corr_map = np.zeros((len(ds.lat), len(ds.lon)))
            for i in range(len(ds.lat)):
                for j in range(len(ds.lon)):
                    if 'time' in ds.dims:
                        ts1 = ds[var1].isel(lat=i, lon=j).values
                        ts2 = ds[var2].isel(lat=i, lon=j).values
                    else:
                        ts1 = ds[var1].isel(lat=i, lon=j).values
                        ts2 = ds[var2].isel(lat=i, lon=j).values
                    
                    if not (np.all(np.isnan(ts1)) or np.all(np.isnan(ts2))):
                        corr_map[i, j] = np.corrcoef(ts1, ts2)[0, 1]
                    else:
                        corr_map[i, j] = np.nan
            
            im = ax2.imshow(corr_map, cmap='RdBu_r', vmin=-1, vmax=1,
                           extent=[ds.lon.min(), ds.lon.max(), 
                                   ds.lat.min(), ds.lat.max()],
                           aspect='auto', origin='lower')
            
            ax2.set_xlabel('Longitude', fontsize=12)
            ax2.set_ylabel('Latitude', fontsize=12)
            ax2.set_title(f'Spatial Correlation Map', fontsize=14, pad=15)
            
            plt.colorbar(im, ax=ax2, label='Correlation Coefficient')
        
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            filename = f"correlation_{var1}_{var2}.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ 相关性图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def create_dashboard(self, ds, variable_name, save_fig=True):
        """
        创建综合仪表板
        
        Parameters:
        -----------
        ds : xarray.Dataset
            数据集
        variable_name : str
            主要变量名
        save_fig : bool
            是否保存图像
        """
        print(f"创建 {variable_name} 数据仪表板...")
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 空间分布
        ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
        if 'time' in ds.dims:
            data = ds[variable_name].isel(time=0)
        else:
            data = ds[variable_name]
        
        im = data.plot(ax=ax1, transform=ccrs.PlateCarree(),
                      cmap='RdBu_r', add_colorbar=False)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax1.set_title(f'{variable_name} Spatial Distribution', fontsize=12)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax1, orientation='horizontal', 
                    shrink=0.8, pad=0.05,
                    label=f'{variable_name} ({data.attrs.get("units", "")})')
        
        # 2. 时间序列（全局平均）
        ax2 = plt.subplot(2, 2, 2)
        ts_global = ds[variable_name].mean(dim=['lat', 'lon']).to_pandas()
        ax2.plot(ts_global.index, ts_global.values, linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel(f'{variable_name} ({data.attrs.get("units", "")})')
        ax2.set_title('Global Mean Time Series')
        ax2.grid(True, alpha=0.3)
        
        # 3. 纬向平均
        ax3 = plt.subplot(2, 2, 3)
        if 'time' in ds.dims:
            zonal_mean = ds[variable_name].mean(dim=['lon', 'time'])
        else:
            zonal_mean = ds[variable_name].mean(dim='lon')
        
        ax3.plot(zonal_mean.lat, zonal_mean.values, linewidth=2)
        ax3.set_xlabel('Latitude')
        ax3.set_ylabel(f'{variable_name} ({data.attrs.get("units", "")})')
        ax3.set_title('Zonal Mean')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # 4. 直方图
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(data.values.flatten(), bins=50, density=True, 
                alpha=0.6, edgecolor='black')
        ax4.set_xlabel(f'{variable_name} ({data.attrs.get("units", "")})')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 添加正态分布曲线
        from scipy import stats as sp_stats
        mu, std = data.mean().values, data.std().values
        x = np.linspace(data.min().values, data.max().values, 100)
        p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax4.plot(x, p, 'r-', linewidth=2, label=f'Normal\nμ={mu:.2f}, σ={std:.2f}')
        ax4.legend()
        
        plt.suptitle(f'{variable_name} Data Dashboard', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # 保存图像
        if save_fig:
            filename = f"{variable_name}_dashboard.png"
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ 数据仪表板已保存: {filepath}")
        
        plt.show()
        return fig

# 使用示例
if __name__ == "__main__":
    # 初始化可视化器
    visualizer = MeteorologicalVisualizer()
    
    # 示例：加载数据
    data_path = "data/raw/temperature_20240101_20240107.nc"
    if Path(data_path).exists():
        ds = xr.open_dataset(data_path)
        
        print("开始数据可视化...")
        
        # 1. 绘制空间分布图
        visualizer.plot_spatial_distribution(ds, 'temperature')
        
        # 2. 绘制时间序列图（北京位置）
        visualizer.plot_time_series(ds, 'temperature', location=(39.9, 116.3))
        
        # 3. 创建数据仪表板
        visualizer.create_dashboard(ds, 'temperature')
        
        print("\n✅ 可视化完成！")
    else:
        print(f"❌ 数据文件不存在: {data_path}")