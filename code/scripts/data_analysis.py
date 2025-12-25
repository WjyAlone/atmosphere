# File: ~/meteorology_analysis/code/scripts/data_analysis.py
"""
大气数据分析工作流程 - 修复版本
"""
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats as scipy_stats  # 重命名避免冲突
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MeteorologicalAnalyzer:
    """气象数据分析器"""
    
    def __init__(self, results_dir="analysis/reports"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def basic_statistics(self, ds, variable_name):
        """基本统计分析"""
        print(f"\n{'='*60}")
        print(f"{variable_name} 基本统计分析")
        print('='*60)
        
        if variable_name not in ds.data_vars:
            print(f"❌ 变量 {variable_name} 不存在")
            return None
        
        data = ds[variable_name]
        
        # 计算统计量
        stats_dict = {
            'mean': float(data.mean().values),
            'median': float(data.median().values),
            'std': float(data.std().values),
            'min': float(data.min().values),
            'max': float(data.max().values),
            'q25': float(data.quantile(0.25).values),
            'q75': float(data.quantile(0.75).values),
            'skewness': float(scipy_stats.skew(data.values.flatten())),
            'kurtosis': float(scipy_stats.kurtosis(data.values.flatten()))
        }
        
        # 打印统计结果
        for key, value in stats_dict.items():
            print(f"{key:15}: {value:10.4f}")
        
        # 保存统计结果
        stats_df = pd.DataFrame([stats_dict])
        stats_file = self.results_dir / f"{variable_name}_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        print(f"✅ 统计结果已保存: {stats_file}")
        
        return stats_dict
    
    def temporal_analysis(self, ds, variable_name, location=None):
        """时间序列分析 - 修复版本"""
        print(f"\n{'='*60}")
        print(f"{variable_name} 时间序列分析")
        print('='*60)
        
        if variable_name not in ds.data_vars:
            print(f"❌ 变量 {variable_name} 不存在")
            return None
        
        # 选择特定位置或全局平均
        if location:
            lat_idx = np.abs(ds.lat - location[0]).argmin()
            lon_idx = np.abs(ds.lon - location[1]).argmin()
            ts_data = ds[variable_name].isel(lat=lat_idx, lon=lon_idx)
            location_str = f"({location[0]}°N, {location[1]}°E)"
        else:
            ts_data = ds[variable_name].mean(dim=['lat', 'lon'])
            location_str = "Global Average"
        
        # 转换为pandas Series进行时间序列分析
        ts_series = ts_data.to_pandas()
        
        # 时间序列统计
        print(f"分析位置: {location_str}")
        print(f"时间范围: {ts_series.index[0]} 到 {ts_series.index[-1]}")
        print(f"数据点数: {len(ts_series)}")
        
        # 趋势分析（线性回归）
        x = np.arange(len(ts_series))
        y = ts_series.values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)  # 使用 scipy_stats
        
        print(f"\n趋势分析:")
        print(f"  斜率 (趋势): {slope:.6f} 单位/天")
        print(f"  R²值: {r_value**2:.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  趋势显著性: {'显著' if p_value < 0.05 else '不显著'}")
        
        # 季节性分析
        if len(ts_series) > 30:  # 如果有足够的数据点
            try:
                ts_series.index = pd.to_datetime(ts_series.index)
                monthly_mean = ts_series.resample('M').mean()
                
                if len(monthly_mean) > 0:
                    seasonal_mean = monthly_mean.groupby(monthly_mean.index.month).mean()
                    
                    print(f"\n月平均 {variable_name}:")
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    for i, month in enumerate(months[:len(seasonal_mean)]):
                        print(f"  {month}: {seasonal_mean.iloc[i]:.2f}")
            except Exception as e:
                print(f"  季节性分析跳过: {e}")
        
        # 保存时间序列数据
        ts_file = self.results_dir / f"{variable_name}_timeseries.csv"
        ts_series.to_csv(ts_file)
        print(f"✅ 时间序列数据已保存: {ts_file}")
        
        return ts_series
    
    def spatial_analysis(self, ds, variable_name, time_idx=0):
        """空间分布分析"""
        print(f"\n{'='*60}")
        print(f"{variable_name} 空间分布分析 (时间索引: {time_idx})")
        print('='*60)
        
        if variable_name not in ds.data_vars:
            print(f"❌ 变量 {variable_name} 不存在")
            return None
        
        # 选择特定时间或时间平均
        if 'time' in ds.dims:
            if time_idx >= len(ds.time):
                print(f"⚠️  时间索引 {time_idx} 超出范围，使用第一个时间点")
                time_idx = 0
            spatial_data = ds[variable_name].isel(time=time_idx)
            time_str = f"Time: {ds.time.isel(time=time_idx).values}"
        else:
            spatial_data = ds[variable_name]
            time_str = "Time Average"
        
        print(time_str)
        
        # 空间统计
        print(f"\n空间统计:")
        print(f"  空间平均值: {float(spatial_data.mean().values):.4f}")
        print(f"  空间标准差: {float(spatial_data.std().values):.4f}")
        
        if float(spatial_data.mean().values) != 0:
            cv = float(spatial_data.std().values / spatial_data.mean().values)
            print(f"  空间变异系数: {cv:.4f}")
        
        # 纬向平均（zonal mean）
        if 'lat' in spatial_data.dims and 'lon' in spatial_data.dims:
            zonal_mean = spatial_data.mean(dim='lon')
            if len(zonal_mean) > 0:
                max_idx = zonal_mean.argmax().values
                min_idx = zonal_mean.argmin().values
                print(f"\n纬向平均:")
                print(f"  最大值在纬度: {float(zonal_mean.lat[max_idx].values):.1f}°")
                print(f"  最小值在纬度: {float(zonal_mean.lat[min_idx].values):.1f}°")
        
        # 经向平均（meridional mean）
        if 'lat' in spatial_data.dims and 'lon' in spatial_data.dims:
            meridional_mean = spatial_data.mean(dim='lat')
            if len(meridional_mean) > 0:
                max_idx = meridional_mean.argmax().values
                min_idx = meridional_mean.argmin().values
                print(f"\n经向平均:")
                print(f"  最大值在经度: {float(meridional_mean.lon[max_idx].values):.1f}°")
                print(f"  最小值在经度: {float(meridional_mean.lon[min_idx].values):.1f}°")
        
        return spatial_data
    
    def correlation_analysis(self, ds, var1, var2):
        """相关性分析"""
        print(f"\n{'='*60}")
        print(f"相关性分析: {var1} vs {var2}")
        print('='*60)
        
        if var1 not in ds.data_vars or var2 not in ds.data_vars:
            print(f"❌ 变量 {var1} 或 {var2} 不存在")
            return None
        
        # 展平数据
        data1_flat = ds[var1].values.flatten()
        data2_flat = ds[var2].values.flatten()
        
        # 移除NaN值
        mask = ~(np.isnan(data1_flat) | np.isnan(data2_flat))
        data1 = data1_flat[mask]
        data2 = data2_flat[mask]
        
        if len(data1) < 2:
            print("❌ 有效数据点不足")
            return None
        
        # 计算相关系数
        corr_coef = np.corrcoef(data1, data2)[0, 1]
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(data1, data2)
        
        print(f"相关系数 (Pearson): {corr_coef:.4f}")
        print(f"R²值: {r_value**2:.4f}")
        print(f"P值: {p_value:.4f}")
        print(f"显著性: {'显著' if p_value < 0.05 else '不显著'}")
        print(f"回归方程: y = {slope:.4f}x + {intercept:.4f}")
        
        # 保存相关分析结果
        corr_results = {
            'variable1': var1,
            'variable2': var2,
            'correlation_coefficient': corr_coef,
            'r_squared': r_value**2,
            'p_value': p_value,
            'slope': slope,
            'intercept': intercept,
            'n_samples': len(data1)
        }
        
        corr_df = pd.DataFrame([corr_results])
        corr_file = self.results_dir / f"correlation_{var1}_{var2}.csv"
        corr_df.to_csv(corr_file, index=False)
        print(f"✅ 相关性分析结果已保存: {corr_file}")
        
        return corr_results

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = MeteorologicalAnalyzer()
    
    # 示例：加载数据
    data_path = "data/raw/temperature_20240101_20240107.nc"
    from pathlib import Path
    
    if Path(data_path).exists():
        ds = xr.open_dataset(data_path)
        
        print("开始大气数据分析...")
        
        # 1. 基本统计分析
        stats = analyzer.basic_statistics(ds, 'temperature')
        
        # 2. 时间序列分析（选择北京位置：39.9°N, 116.3°E）
        beijing_loc = (39.9, 116.3)
        ts_data = analyzer.temporal_analysis(ds, 'temperature', beijing_loc)
        
        # 3. 空间分析
        spatial_data = analyzer.spatial_analysis(ds, 'temperature')
        
        print("\n✅ 分析完成！")
    else:
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行 data_acquisition.py 创建示例数据")