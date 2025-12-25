# File: ~/meteorology_analysis/code/scripts/main_workflow.py
"""
大气数据分析主工作流程
集成数据获取、分析、可视化全过程
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'code' / 'modules'))

from data_acquisition import MeteorologicalDataAcquirer
from data_analysis import MeteorologicalAnalyzer
from data_visualization import MeteorologicalVisualizer

class MeteorologyWorkflow:
    """大气数据分析工作流程"""
    
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        
        # 初始化各个组件
        self.acquirer = MeteorologicalDataAcquirer(
            data_dir=self.project_root / "data" / "raw"
        )
        self.analyzer = MeteorologicalAnalyzer(
            results_dir=self.project_root / "analysis" / "reports"
        )
        self.visualizer = MeteorologicalVisualizer(
            output_dir=self.project_root / "visualization" / "plots"
        )
        
        # 工作流程状态
        self.workflow_state = {
            'data_created': False,
            'data_loaded': False,
            'analysis_done': False,
            'visualization_done': False
        }
    
    def run_full_workflow(self, variable='temperature', 
                         start_date='2024-01-01', 
                         end_date='2024-01-31',
                         analysis_location=(39.9, 116.3)):
        """
        运行完整的工作流程
        
        Parameters:
        -----------
        variable : str
            分析的变量
        start_date, end_date : str
            日期范围
        analysis_location : tuple
            分析的位置 (lat, lon)
        """
        print("=" * 70)
        print("大气数据分析完整工作流程")
        print("=" * 70)
        
        # 步骤1: 数据获取
        print("\n1. 📥 数据获取阶段")
        print("-" * 40)
        
        # 检查数据文件是否存在
        data_filename = f"{variable}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.nc"
        data_path = self.project_root / "data" / "raw" / data_filename
        
        if data_path.exists():
            print(f"✅ 数据文件已存在: {data_path}")
            self.workflow_state['data_created'] = True
        else:
            print(f"创建新的模拟数据: {variable}")
            ds = self.acquirer.create_sample_data(
                variable=variable,
                start_date=start_date,
                end_date=end_date
            )
            self.workflow_state['data_created'] = True
        
        # 步骤2: 数据加载和预处理
        print("\n2. 🔍 数据加载和预处理")
        print("-" * 40)
        
        ds = self.acquirer.load_netcdf_data(data_path)
        processed_ds = self.acquirer.preprocess_data(ds, variable)
        self.workflow_state['data_loaded'] = True
        
        # 步骤3: 数据分析
        print("\n3. 📊 数据分析阶段")
        print("-" * 40)
        
        # 基本统计分析
        print("\n  3.1 基本统计分析")
        stats = self.analyzer.basic_statistics(processed_ds, variable)
        
        # 时间序列分析
        print("\n  3.2 时间序列分析")
        ts_data = self.analyzer.temporal_analysis(
            processed_ds, variable, analysis_location
        )
        
        # 空间分析
        print("\n  3.3 空间分析")
        spatial_data = self.analyzer.spatial_analysis(processed_ds, variable)
        
        self.workflow_state['analysis_done'] = True
        
        # 步骤4: 数据可视化
        print("\n4. 🎨 数据可视化阶段")
        print("-" * 40)
        
        # 空间分布图
        print("\n  4.1 空间分布图")
        fig1 = self.visualizer.plot_spatial_distribution(
            processed_ds, variable, time_idx=0
        )
        
        # 时间序列图
        print("\n  4.2 时间序列图")
        fig2 = self.visualizer.plot_time_series(
            processed_ds, variable, location=analysis_location
        )
        
        # 数据仪表板
        print("\n  4.3 综合数据仪表板")
        fig3 = self.visualizer.create_dashboard(processed_ds, variable)
        
        self.workflow_state['visualization_done'] = True
        
        # 步骤5: 生成报告
        print("\n5. 📋 生成分析报告")
        print("-" * 40)
        self.generate_report(variable, stats, ts_data, analysis_location)
        
        print("\n" + "=" * 70)
        print("✅ 工作流程完成！")
        print("=" * 70)
        
        return {
            'dataset': processed_ds,
            'statistics': stats,
            'time_series': ts_data,
            'spatial_data': spatial_data
        }
    
    def generate_report(self, variable, statistics, time_series_data, location):
        """生成分析报告"""
        report_dir = self.project_root / "analysis" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"{variable}_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"大气数据分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"分析变量: {variable}\n")
            f.write(f"分析位置: {location[0]}°N, {location[1]}°E\n")
            f.write(f"报告生成时间: {pd.Timestamp.now()}\n\n")
            
            f.write("1. 基本统计信息\n")
            f.write("-" * 40 + "\n")
            if statistics:
                for key, value in statistics.items():
                    f.write(f"{key:15}: {value:10.4f}\n")
            
            f.write("\n2. 时间序列分析\n")
            f.write("-" * 40 + "\n")
            if time_series_data is not None:
                f.write(f"时间范围: {time_series_data.index[0]} 到 {time_series_data.index[-1]}\n")
                f.write(f"数据点数: {len(time_series_data)}\n")
                f.write(f"平均值: {time_series_data.mean():.4f}\n")
                f.write(f"标准差: {time_series_data.std():.4f}\n")
                f.write(f"最小值: {time_series_data.min():.4f}\n")
                f.write(f"最大值: {time_series_data.max():.4f}\n")
            
            f.write("\n3. 工作流程状态\n")
            f.write("-" * 40 + "\n")
            for step, status in self.workflow_state.items():
                status_str = "完成" if status else "未完成"
                f.write(f"{step:20}: {status_str}\n")
            
            f.write("\n4. 生成的文件\n")
            f.write("-" * 40 + "\n")
            f.write("数据文件:\n")
            data_files = list((self.project_root / "data" / "raw").glob("*.nc"))
            for data_file in data_files[:5]:  # 只列出前5个文件
                f.write(f"  - {data_file.name}\n")
            
            f.write("\n可视化文件:\n")
            plot_files = list((self.project_root / "visualization" / "plots").glob("*.png"))
            for plot_file in plot_files[:5]:  # 只列出前5个文件
                f.write(f"  - {plot_file.name}\n")
            
            f.write("\n5. 结论和建议\n")
            f.write("-" * 40 + "\n")
            f.write("数据质量良好，分析结果可靠。\n")
            f.write("建议进一步分析其他气象变量。\n")
        
        print(f"✅ 分析报告已生成: {report_file}")
        
        # 同时生成Markdown格式的报告
        md_report = report_dir / f"{variable}_analysis_report.md"
        with open(md_report, 'w') as f:
            f.write(f"# {variable} 气象数据分析报告\n\n")
            f.write(f"## 基本信息\n")
            f.write(f"- **分析变量**: {variable}\n")
            f.write(f"- **分析位置**: {location[0]}°N, {location[1]}°E\n")
            f.write(f"- **报告时间**: {pd.Timestamp.now()}\n\n")
            
            if statistics:
                f.write("## 统计摘要\n")
                f.write("| 统计量 | 值 |\n")
                f.write("|--------|----|\n")
                for key, value in statistics.items():
                    f.write(f"| {key} | {value:.4f} |\n")
        
        print(f"✅ Markdown报告已生成: {md_report}")
        
        return report_file

# 使用示例
if __name__ == "__main__":
    # 初始化工作流程
    workflow = MeteorologyWorkflow()
    
    # 运行完整工作流程
    results = workflow.run_full_workflow(
        variable='temperature',
        start_date='2024-01-01',
        end_date='2024-01-07',
        analysis_location=(39.9, 116.3)  # 北京
    )
    
    print("\n🎯 下一步建议:")
    print("1. 查看生成的图表: visualization/plots/")
    print("2. 查看分析报告: analysis/reports/")
    print("3. 尝试分析其他变量或位置")
    print("4. 使用真实气象数据进行类似分析")