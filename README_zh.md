自动化硬度分级评估系统
基于无监督学习的自动化硬度分级评估系统，利用按压实验采集的力矩、方向和Paxini传感器数据，自动评估被测点硬度并生成9×11硬度网格。

系统概述
本系统通过分析按压实验数据（包括六维力传感器数据、位置数据和239个Paxini触点数据），使用无监督学习算法自动将硬度分为4个等级，并为每个样本生成可视化的9×11硬度分数网格。

主要特性
无监督学习：无需预先标注数据，自动发现硬度模式

多特征融合：结合力学特征、Paxini统计特征和力矩特征

智能聚类：基于刚度特征自动映射有意义的硬度等级

网格可视化：生成直观的9×11硬度分布网格

实时预测：支持实时数据监控和预测

批量处理：一次性处理多个样本文件

系统架构

自动化硬度分级评估系统/
├── core_processor.py     # 核心数据处理和模型训练
├── main.py              # 主控制程序
├── realtime_predictor.py # 实时预测模块
├── config.py            # 配置文件
├── data92/
│   └── data926/         # 数据文件目录（12个CSV样本）
├── models/              # 训练模型保存目录
├── results/             # 输出结果目录
└── 指腹L5325 PX6AX-GEN3-CP-L5325-Omega PXSR-STDCP03A.xlsx  # 坐标文件

数据格式
每个CSV文件包含：

前6列：六维力传感器数据（XYZ分力 + XYZ扭矩）

接着3列：位置数据

接着4列：四元数

剩余239列：Paxini传感器触点数据

安装要求

pip install pandas numpy scipy scikit-learn matplotlib openpyxl

使用方法
1. 环境检查

python main.py
选择选项4：检查数据环境

2. 离线训练

选择选项1：离线训练模型

系统将：

处理所有CSV样本文件

提取特征并训练聚类模型

生成硬度等级和9×11网格

保存模型和可视化结果

3. 预测模式
选项2：实时预测（监控新文件）

选项3：批量预测所有文件

输出结果
训练完成后，在results目录中生成：

hardness_assessment_results.csv - 样本硬度等级

{样本名}_hardness_grid_9x11.csv - 每个样本的9×11网格

all_samples_hardness_grids.png - 所有样本可视化

feature_importance.png - 特征重要性排名

clustering_info.txt - 聚类详细信息

配置参数
在config.py中调整：

NUM_CLUSTERS: 硬度等级数量（默认4）

数据路径和文件位置

传感器数据列索引

可视化参数

技术细节
算法: KMeans聚类 + 特征标准化

特征工程: 统计特征提取（避免维度灾难）

硬度映射: 基于刚度特征的智能重映射

网格生成: 基于坐标插值的9×11规则网格

许可证
MIT License