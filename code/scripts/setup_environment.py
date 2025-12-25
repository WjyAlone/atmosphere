# File: ~/meteorology_analysis/code/scripts/setup_environment.py
"""
大气数据分析环境设置脚本
"""
import sys
import subprocess
import os

def check_environment():
    """检查Python环境和必要包"""
    print("=" * 60)
    print("大气数据分析环境检查")
    print("=" * 60)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查工作目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 建议的工作目录结构
    recommended_dirs = [
        'data/raw',
        'data/processed', 
        'code/scripts',
        'code/modules',
        'analysis/notebooks',
        'analysis/reports',
        'visualization/plots',
        'visualization/maps'
    ]
    
    print("\n建议的目录结构:")
    for dir_path in recommended_dirs:
        full_path = os.path.join(current_dir, dir_path)
        exists = "✓" if os.path.exists(full_path) else "✗"
        print(f"  {exists} {dir_path}")
    
    return True

def setup_project_structure(base_path="."):
    """设置项目目录结构"""
    directories = [
        'data/raw',
        'data/processed',
        'data/interim',
        'code/scripts',
        'code/modules',
        'code/config',
        'analysis/notebooks',
        'analysis/scripts',
        'analysis/reports',
        'visualization/plots',
        'visualization/maps',
        'visualization/animation',
        'docs',
        'outputs',
        'logs'
    ]
    
    print("\n创建项目目录结构...")
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"  创建: {dir_path}")
        
        # 在每个目录中创建 .gitkeep 文件（如果是Git项目）
        gitkeep_file = os.path.join(full_path, '.gitkeep')
        with open(gitkeep_file, 'w') as f:
            pass
    
    print("✅ 项目目录结构创建完成")
    return True

if __name__ == "__main__":
    check_environment()
    response = input("\n是否创建项目目录结构? (y/n): ")
    if response.lower() == 'y':
        setup_project_structure()