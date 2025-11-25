import os

# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 数据路径配置
    'DATA_DIR': os.path.join(BASE_DIR, 'data92', 'data926'),
    'COORDINATES_FILE': os.path.join(BASE_DIR, '指腹L5325 PX6AX-GEN3-CP-L5325-Omega PXSR-STDCP03A.xlsx'),
    'OUTPUT_DIR': os.path.join(BASE_DIR, 'results'),
    'MODEL_DIR': os.path.join(BASE_DIR, 'models'),
    
    # 聚类配置 - 由于只有12个样本，调整为4个等级
    'NUM_CLUSTERS': 4,
    'RANDOM_STATE': 42,
    
    # 数据列配置
    'FORCE_Z_INDEX': 2,
    'POSITION_Z_INDEX': 8,
    'TORQUE_INDICES': [3, 4, 5],
    'PAXINI_START_INDEX': 13,
    'PAXINI_NUM_POINTS': 239,
    
    # 可视化配置
    'CHINESE_FONT': 'SimHei',
    'FIGURE_SIZE': (14, 10),
    
    # 网格配置
    'GRID_SHAPE': (9, 11),
    
    # 实时预测配置
    'REALTIME_UPDATE_INTERVAL': 1000,
}

# 创建必要的目录
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

FEATURE_NAMES = {
    'stiffness': '刚度K',
    'max_force': '最大力值',
    'work_done': '做功',
    'force_range': '力范围',
    'paxini_mean': 'Paxini均值',
    'paxini_std': 'Paxini标准差',
    'paxini_max': 'Paxini最大值',
    'torque_x': 'X方向力矩',
    'torque_y': 'Y方向力矩', 
    'torque_z': 'Z方向力矩'
}