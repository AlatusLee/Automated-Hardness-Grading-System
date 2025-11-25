import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core_processor import HardnessProcessor
from realtime_predictor import RealTimePredictor
from config import CONFIG
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def offline_training():
    """离线训练模型"""
    print("=== 离线硬度分级模型训练 ===")
    
    # 初始化处理器
    processor = HardnessProcessor()
    
    # 处理数据
    if not processor.process_all_files():
        print("数据处理失败，请检查数据和配置")
        return
    
    # 训练聚类模型
    clustering_info = processor.train_clustering_model()
    if clustering_info is None:
        print("模型训练失败")
        return
    
    hardness_scores = clustering_info['labels']
    
    # 为每个样本生成硬度网格
    grid_scores_dict = {}
    for filename in processor.file_names:
        file_path = os.path.join(CONFIG['DATA_DIR'], filename)
        grid_scores = processor.create_hardness_grid_for_sample(file_path)
        if grid_scores is not None:
            grid_scores_dict[filename] = grid_scores
    
    print(f"成功生成 {len(grid_scores_dict)} 个硬度网格")
    
    # 保存模型和结果
    model_path = os.path.join(CONFIG['MODEL_DIR'], 'hardness_model.pkl')
    processor.save_model(model_path)
    
    # 保存结果
    processor.save_results(hardness_scores, grid_scores_dict, clustering_info)
    
    # 可视化结果
    visualize_results(processor, hardness_scores, grid_scores_dict)
    
    print("离线训练完成！")

def visualize_results(processor, hardness_scores, grid_scores_dict):
    """可视化结果"""
    plt.rcParams['font.sans-serif'] = [CONFIG['CHINESE_FONT']]
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建多个子图来显示所有样本
    n_samples = len(processor.file_names)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (filename, hardness) in enumerate(zip(processor.file_names, hardness_scores)):
        if i >= len(axes):
            break
            
        if filename in grid_scores_dict:
            grid_scores = grid_scores_dict[filename]
            im = axes[i].imshow(grid_scores, cmap='viridis', origin='lower')
            axes[i].set_title(f'{filename}\n硬度等级: {hardness + 1}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    # 隐藏多余的子图
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'all_samples_hardness_grids.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 显示特征重要性（基于聚类中心距离）
    visualize_feature_importance(processor, clustering_info)

def visualize_feature_importance(processor, clustering_info):
    """可视化特征重要性"""
    try:
        # 计算特征对聚类中心的方差贡献
        cluster_centers = processor.cluster_model.cluster_centers_
        feature_importance = np.std(cluster_centers, axis=0)
        
        # 排序
        sorted_indices = np.argsort(feature_importance)[::-1]
        sorted_features = [processor.feature_names[i] for i in sorted_indices]
        sorted_importance = feature_importance[sorted_indices]
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(sorted_features))
        
        plt.barh(y_pos, sorted_importance, align='center', alpha=0.7)
        plt.yticks(y_pos, sorted_features)
        plt.xlabel('特征重要性（聚类中心标准差）')
        plt.title('硬度分级特征重要性排名')
        
        # 在条形上添加数值
        for i, v in enumerate(sorted_importance):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['OUTPUT_DIR'], 'feature_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"可视化特征重要性失败: {e}")

def realtime_prediction():
    """实时预测"""
    print("=== 实时硬度预测 ===")
    
    predictor = RealTimePredictor()
    
    print("\n请选择预测模式:")
    print("1. 实时监控模式（自动检测新文件）")
    print("2. 单文件预测模式")
    
    choice = input("请选择模式 (1-2): ").strip()
    
    if choice == '1':
        print("\n启动实时监控模式...")
        predictor.start_realtime_monitoring()
    elif choice == '2':
        print("\n启动单文件预测模式...")
        predictor.predict_single_file_interactive()
    else:
        print("无效选择")

def check_data():
    """检查数据环境"""
    print("=== 数据环境检查 ===")
    
    # 检查数据目录
    if not os.path.exists(CONFIG['DATA_DIR']):
        print(f"[错误] 数据目录不存在: {CONFIG['DATA_DIR']}")
        return False
    else:
        print(f"[成功] 数据目录存在: {CONFIG['DATA_DIR']}")
    
    # 检查坐标文件
    if not os.path.exists(CONFIG['COORDINATES_FILE']):
        print(f"[错误] 坐标文件不存在: {CONFIG['COORDINATES_FILE']}")
        return False
    else:
        print(f"[成功] 坐标文件存在: {CONFIG['COORDINATES_FILE']}")
    
    # 检查CSV文件
    csv_files = [f for f in os.listdir(CONFIG['DATA_DIR']) if f.endswith('.csv')]
    if not csv_files:
        print(f"[错误] 在 {CONFIG['DATA_DIR']} 中没有找到CSV文件")
        return False
    else:
        print(f"[成功] 找到 {len(csv_files)} 个CSV文件")
        print("所有文件:")
        for i, file in enumerate(csv_files):
            print(f"  {i+1}. {file}")
    
    # 检查模型目录
    if not os.path.exists(CONFIG['MODEL_DIR']):
        print(f"[警告] 模型目录不存在，将自动创建")
        os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
    else:
        print(f"[成功] 模型目录存在: {CONFIG['MODEL_DIR']}")
        
        # 检查是否有训练好的模型
        model_path = os.path.join(CONFIG['MODEL_DIR'], 'hardness_model.pkl')
        if os.path.exists(model_path):
            print(f"[成功] 找到训练好的模型: {model_path}")
        else:
            print(f"[警告] 未找到训练好的模型，请先运行离线训练")
    
    # 检查输出目录
    if not os.path.exists(CONFIG['OUTPUT_DIR']):
        print(f"[警告] 输出目录不存在，将自动创建")
        os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
    else:
        print(f"[成功] 输出目录存在: {CONFIG['OUTPUT_DIR']}")
    
    print("\n数据环境检查完成!")
    return True

def batch_prediction():
    """批量预测所有文件"""
    print("=== 批量预测所有文件 ===")
    
    processor = HardnessProcessor()
    
    # 加载模型
    model_path = os.path.join(CONFIG['MODEL_DIR'], 'hardness_model.pkl')
    if not processor.load_model(model_path):
        print("模型加载失败，请先运行离线训练")
        return
    
    csv_files = [f for f in os.listdir(CONFIG['DATA_DIR']) if f.endswith('.csv')]
    
    results = []
    for filename in csv_files:
        file_path = os.path.join(CONFIG['DATA_DIR'], filename)
        hardness_level, grid_scores, features = processor.predict_single_file(file_path)
        
        if hardness_level is not None:
            results.append({
                'file_name': filename,
                'hardness_level': hardness_level + 1,
                'grid_scores': grid_scores
            })
            print(f"文件: {filename} -> 硬度等级: {hardness_level + 1}")
            
            # 保存该文件的网格
            base_name = os.path.splitext(filename)[0]
            grid_path = os.path.join(CONFIG['OUTPUT_DIR'], f'{base_name}_predicted_grid.csv')
            pd.DataFrame(grid_scores).to_csv(grid_path, index=False, header=False)
        else:
            print(f"文件: {filename} -> 预测失败")
    
    # 保存批量预测结果
    if results:
        results_df = pd.DataFrame([{
            'file_name': r['file_name'],
            'hardness_level': r['hardness_level']
        } for r in results])
        results_path = os.path.join(CONFIG['OUTPUT_DIR'], 'batch_prediction_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"\n批量预测完成！结果已保存到: {results_path}")

def main():
    """主控制函数"""
    while True:
        print("\n" + "="*50)
        print("          自动化硬度分级评估系统")
        print("="*50)
        print("1. 离线训练模型")
        print("2. 实时预测")
        print("3. 批量预测所有文件")
        print("4. 检查数据")
        print("5. 退出")
        print("="*50)
        
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == '1':
            print("\n开始离线训练...")
            offline_training()
        elif choice == '2':
            print("\n开始实时预测...")
            realtime_prediction()
        elif choice == '3':
            print("\n开始批量预测...")
            batch_prediction()
        elif choice == '4':
            check_data()
        elif choice == '5':
            print("感谢使用！再见！")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()