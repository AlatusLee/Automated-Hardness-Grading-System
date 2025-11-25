import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core_processor import HardnessProcessor
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

class RealTimePredictor:
    def __init__(self):
        self.processor = HardnessProcessor()
        self.model_loaded = False
        self.fig = None
        self.ax = None
        self.im = None
        self.current_hardness = None
        self.current_grid = None
        
    def load_model(self):
        """加载预训练模型"""
        model_path = os.path.join(CONFIG['MODEL_DIR'], 'hardness_model.pkl')
        if not os.path.exists(model_path):
            print("未找到预训练模型，请先运行离线训练")
            return False
        
        if self.processor.load_model(model_path):
            self.model_loaded = True
            print("模型加载成功")
            return True
        else:
            print("模型加载失败")
            return False
    
    def setup_visualization(self):
        """设置实时可视化"""
        plt.rcParams['font.sans-serif'] = [CONFIG['CHINESE_FONT']]
        plt.rcParams['axes.unicode_minus'] = False
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # 初始化网格图
        self.current_grid = np.zeros(CONFIG['GRID_SHAPE'])
        self.im = self.ax.imshow(self.current_grid, cmap='viridis', origin='lower')
        
        self.ax.set_title('实时硬度网格预测', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        
        # 添加颜色条
        plt.colorbar(self.im, ax=self.ax)
        self.ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 信息显示
        self.info_text = self.ax.text(
            0.02, 0.98, '准备就绪',
            transform=self.ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10
        )
        
        plt.tight_layout()
    
    def get_latest_data_file(self):
        """获取最新的数据文件（模拟实时数据）"""
        csv_files = [f for f in os.listdir(CONFIG['DATA_DIR']) if f.endswith('.csv')]
        if not csv_files:
            return None
        
        # 获取最新的文件
        latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(CONFIG['DATA_DIR'], x)))
        return os.path.join(CONFIG['DATA_DIR'], latest_file)
    
    def update_prediction(self, frame):
        """更新预测结果"""
        if not self.model_loaded:
            return
        
        # 获取最新数据文件
        file_path = self.get_latest_data_file()
        if file_path is None:
            return
        
        try:
            # 预测硬度
            hardness_level, grid_scores, features = self.processor.predict_single_file(file_path)
            if hardness_level is not None and grid_scores is not None:
                self.current_hardness = hardness_level
                self.current_grid = grid_scores
                
                # 更新网格图
                self.im.set_data(grid_scores)
                self.im.set_clim(0, 255)  # Paxini值范围
                
                # 更新信息
                filename = os.path.basename(file_path)
                self.info_text.set_text(f'文件: {filename}\n硬度等级: {hardness_level + 1}\n更新时间: {time.strftime("%H:%M:%S")}')
                
                print(f"实时更新 - 文件: {filename}, 硬度等级: {hardness_level + 1}")
                
        except Exception as e:
            logger.error(f"实时更新失败: {e}")
    
    def start_realtime_monitoring(self):
        """开始实时监控"""
        if not self.load_model():
            return
        
        self.setup_visualization()
        
        # 创建动画
        ani = FuncAnimation(
            self.fig, 
            self.update_prediction,
            interval=CONFIG['REALTIME_UPDATE_INTERVAL'],
            cache_frame_data=False
        )
        
        print("开始实时硬度监控...")
        print("系统将自动检测数据目录中的新文件并更新预测结果")
        print("按Ctrl+C退出")
        
        plt.show()
    
    def predict_single_file_interactive(self):
        """交互式单文件预测"""
        if not self.load_model():
            return
        
        csv_files = [f for f in os.listdir(CONFIG['DATA_DIR']) if f.endswith('.csv')]
        if not csv_files:
            print("没有找到CSV文件")
            return
        
        print("\n可用的CSV文件:")
        for i, filename in enumerate(csv_files):
            print(f"{i+1}. {filename}")
        
        try:
            choice = int(input("\n请选择要预测的文件编号: ")) - 1
            if choice < 0 or choice >= len(csv_files):
                print("无效的选择")
                return
            
            file_path = os.path.join(CONFIG['DATA_DIR'], csv_files[choice])
            
            # 预测硬度
            hardness_level, grid_scores, features = self.processor.predict_single_file(file_path)
            if hardness_level is not None and grid_scores is not None:
                self.show_prediction_result(hardness_level, grid_scores, csv_files[choice], features)
            else:
                print("预测失败")
                
        except (ValueError, IndexError):
            print("无效的输入")
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
    
    def show_prediction_result(self, hardness_level, grid_scores, filename, features):
        """显示预测结果"""
        plt.rcParams['font.sans-serif'] = [CONFIG['CHINESE_FONT']]
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左侧：硬度网格
        im = ax1.imshow(grid_scores, cmap='viridis', origin='lower')
        ax1.set_title(f'硬度分布 - {filename}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Y坐标', fontsize=12)
        ax1.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        plt.colorbar(im, ax=ax1)
        
        # 右侧：特征信息
        ax2.axis('off')
        info_text = f'文件: {filename}\n硬度等级: {hardness_level + 1}\n\n关键特征值:\n'
        
        # 显示最重要的几个特征
        important_features = [
            ('stiffness', '刚度'),
            ('paxini_mean', 'Paxini均值'),
            ('work_done', '做功'),
            ('max_force', '最大力值'),
            ('force_range', '力范围')
        ]
        
        for feat_key, feat_name in important_features:
            if feat_key in features:
                value = features[feat_key]
                if isinstance(value, float):
                    info_text += f'{feat_name}: {value:.4f}\n'
                else:
                    info_text += f'{feat_name}: {value}\n'
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()