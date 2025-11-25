import pandas as pd
import numpy as np
import os
import pickle
from scipy import stats
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from config import CONFIG, FEATURE_NAMES
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardnessProcessor:
    def __init__(self):
        self.coordinates = None
        self.feature_matrix = None
        self.file_names = []
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.feature_names = []
        
    def load_coordinates(self):
        """加载坐标数据"""
        try:
            df = pd.read_excel(CONFIG['COORDINATES_FILE'], sheet_name=0)
            self.coordinates = df[['X', 'Y', 'Z']].values
            logger.info(f"成功加载 {len(self.coordinates)} 个坐标点")
            return True
        except Exception as e:
            logger.error(f"加载坐标文件失败: {e}")
            return False
    
    def extract_features_from_file(self, file_path):
        """从单个CSV文件提取特征 - 每个文件一个样本"""
        try:
            filename = os.path.basename(file_path)
            df = pd.read_csv(file_path, header=None, 
                           names=[f'col_{i}' for i in range(730)], 
                           low_memory=False)
            
            # 提取单个样本的特征
            sample_features = self._extract_single_sample_features(df, filename)
            return sample_features
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
            return None
    
    def _extract_single_sample_features(self, df, filename):
        """为单个样本提取特征"""
        try:
            # 基本列配置
            fz_col = f'col_{CONFIG["FORCE_Z_INDEX"]}'
            z_col = f'col_{CONFIG["POSITION_Z_INDEX"]}'
            
            # 数据清洗
            if fz_col not in df.columns or z_col not in df.columns:
                return None
                
            df[fz_col] = pd.to_numeric(df[fz_col], errors='coerce')
            df[z_col] = pd.to_numeric(df[z_col], errors='coerce')
            df.dropna(subset=[fz_col, z_col], inplace=True)
            
            if df.empty:
                return None
            
            # 提取全局特征
            global_features = self._extract_global_features(df, fz_col, z_col)
            if global_features is None:
                return None
            
            # 提取Paxini统计特征
            paxini_features = self._extract_paxini_statistical_features(df)
            
            # 提取力矩特征
            torque_features = self._extract_torque_features(df, global_features['peak_index'])
            
            # 合并所有特征
            all_features = {}
            all_features.update(global_features)
            all_features.update(paxini_features)
            all_features.update(torque_features)
            all_features['file_name'] = filename
            
            return all_features
            
        except Exception as e:
            logger.error(f"提取样本特征失败: {e}")
            return None
    
    def _extract_global_features(self, df, fz_col, z_col):
        """提取全局特征"""
        try:
            # 寻找峰值点
            peak_index = df[fz_col].idxmin()
            peak_point = df.loc[peak_index]
            
            pre_peak_data = df.loc[:peak_index]
            try:
                start_index = pre_peak_data[pre_peak_data[fz_col] < -0.5].index[0]
            except IndexError:
                start_index = df.index[0]
            
            start_point = df.loc[start_index]
            
            # 计算刚度
            delta_fz = peak_point[fz_col] - start_point[fz_col]
            delta_z = start_point[z_col] - peak_point[z_col]
            
            if abs(delta_z) < 1e-9:
                return None
                
            stiffness_K = abs(delta_fz / delta_z)
            
            # 提取力特征
            force_data = df[fz_col].values
            position_data = df[z_col].values
            
            features = {
                'stiffness': stiffness_K,
                'start_force': start_point[fz_col],
                'peak_force': peak_point[fz_col],
                'max_force': np.max(force_data),
                'min_force': np.min(force_data),
                'mean_force': np.mean(force_data),
                'force_range': np.ptp(force_data),
                'force_std': np.std(force_data),
                'peak_index': peak_index
            }
            
            # 计算做功
            if len(force_data) > 1:
                work_done = np.trapz(np.abs(force_data), position_data)
                features['work_done'] = work_done
            else:
                features['work_done'] = 0
                
            return features
            
        except Exception as e:
            logger.error(f"提取全局特征失败: {e}")
            return None
    
    def _extract_paxini_statistical_features(self, df):
        """提取Paxini数据的统计特征"""
        try:
            paxini_values = []
            peak_index = df[f'col_{CONFIG["FORCE_Z_INDEX"]}'].idxmin()
            
            # 收集所有Paxini触点的峰值数据
            for contact_idx in range(CONFIG['PAXINI_NUM_POINTS']):
                paxini_col = f'col_{CONFIG["PAXINI_START_INDEX"] + contact_idx}'
                if paxini_col in df.columns:
                    value = pd.to_numeric(df.loc[peak_index, paxini_col], errors='coerce')
                    if not np.isnan(value):
                        paxini_values.append(value)
            
            if not paxini_values:
                return {}
            
            # 计算统计特征
            paxini_array = np.array(paxini_values)
            
            features = {
                'paxini_mean': np.mean(paxini_array),
                'paxini_std': np.std(paxini_array),
                'paxini_max': np.max(paxini_array),
                'paxini_min': np.min(paxini_array),
                'paxini_range': np.ptp(paxini_array),
                'paxini_median': np.median(paxini_array),
                'paxini_q25': np.percentile(paxini_array, 25),
                'paxini_q75': np.percentile(paxini_array, 75),
            }
            
            # 添加分布特征
            if len(paxini_array) > 1:
                try:
                    features['paxini_skew'] = stats.skew(paxini_array)
                    features['paxini_kurtosis'] = stats.kurtosis(paxini_array)
                except:
                    features['paxini_skew'] = 0
                    features['paxini_kurtosis'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"提取Paxini统计特征失败: {e}")
            return {}
    
    def _extract_torque_features(self, df, peak_index):
        """提取力矩特征"""
        try:
            torque_features = {}
            torque_cols = [f'col_{i}' for i in CONFIG['TORQUE_INDICES']]
            
            for i, col in enumerate(torque_cols):
                if col in df.columns:
                    torque_value = df.loc[peak_index, col]
                    numeric_value = pd.to_numeric(torque_value, errors='coerce')
                    if not np.isnan(numeric_value):
                        torque_features[f'torque_{i}'] = numeric_value
            
            return torque_features
        except Exception as e:
            logger.error(f"提取力矩特征失败: {e}")
            return {}
    
    def process_all_files(self):
        """处理所有数据文件"""
        if not self.load_coordinates():
            return False
            
        csv_files = [f for f in os.listdir(CONFIG['DATA_DIR']) if f.endswith('.csv')]
        
        if not csv_files:
            logger.error(f"在 {CONFIG['DATA_DIR']} 中没有找到CSV文件")
            return False
            
        logger.info(f"找到 {len(csv_files)} 个CSV文件，开始处理...")
        
        all_sample_features = []
        
        for filename in csv_files:
            file_path = os.path.join(CONFIG['DATA_DIR'], filename)
            sample_features = self.extract_features_from_file(file_path)
            
            if sample_features:
                all_sample_features.append(sample_features)
                self.file_names.append(filename)
        
        if len(all_sample_features) == 0:
            logger.error("没有成功提取任何特征")
            return False
        
        # 构建特征矩阵 - 每个文件一个样本
        self._build_feature_matrix(all_sample_features)
        
        logger.info(f"成功处理 {len(all_sample_features)} 个样本，特征维度: {self.feature_matrix.shape}")
        logger.info(f"特征列表: {self.feature_names}")
        
        return True
    
    def _build_feature_matrix(self, all_sample_features):
        """构建特征矩阵"""
        # 找出所有特征键（排除非数值字段）
        feature_keys = set()
        for features in all_sample_features:
            for key in features.keys():
                if key not in ['file_name', 'peak_index'] and isinstance(features[key], (int, float)):
                    feature_keys.add(key)
        
        self.feature_names = sorted(feature_keys)
        
        # 构建矩阵
        self.feature_matrix = np.array([[features.get(key, 0) for key in self.feature_names] 
                                      for features in all_sample_features])
    
    def train_clustering_model(self):
        """训练聚类模型"""
        if self.feature_matrix is None:
            logger.error("特征矩阵未构建")
            return None
        
        try:
            # 数据标准化
            features_scaled = self.scaler.fit_transform(self.feature_matrix)
            
            # 调整聚类数量，确保不超过样本数
            n_samples = len(self.feature_matrix)
            n_clusters = min(CONFIG['NUM_CLUSTERS'], n_samples - 1)
            if n_clusters < 2:
                logger.error("样本数量太少，无法进行聚类")
                return None
            
            # KMeans聚类
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=CONFIG['RANDOM_STATE'],
                n_init=10
            )
            labels = self.cluster_model.fit_predict(features_scaled)
            
            # 计算轮廓系数
            if len(set(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
            else:
                score = -1
                
            logger.info(f"聚类完成，使用 {n_clusters} 个聚类，轮廓系数: {score:.4f}")
            
            # 根据刚度重新映射标签（刚度越大，硬度等级越高）
            remapped_labels = self._remap_labels_by_stiffness(labels)
            
            clustering_info = {
                'algorithm': 'KMeans',
                'optimal_clusters': n_clusters,
                'silhouette_score': score,
                'labels': remapped_labels,
                'original_labels': labels
            }
            
            return clustering_info
            
        except Exception as e:
            logger.error(f"训练聚类模型失败: {e}")
            return None
    
    def _remap_labels_by_stiffness(self, labels):
        """根据刚度特征重新映射标签，使刚度越大硬度等级越高"""
        try:
            # 找到刚度特征在特征名中的索引
            stiffness_idx = None
            for i, name in enumerate(self.feature_names):
                if 'stiffness' in name:
                    stiffness_idx = i
                    break
            
            if stiffness_idx is None:
                logger.warning("未找到刚度特征，使用原始标签")
                return labels
            
            # 计算每个聚类的平均刚度
            cluster_stiffness = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                mask = (labels == label)
                avg_stiffness = np.mean(self.feature_matrix[mask, stiffness_idx])
                cluster_stiffness.append((label, avg_stiffness))
            
            # 按刚度排序（从低到高）
            cluster_stiffness.sort(key=lambda x: x[1])
            
            # 创建映射：刚度最低的为1级，最高的为n_clusters级
            label_mapping = {}
            for hardness_level, (original_label, _) in enumerate(cluster_stiffness, 1):
                label_mapping[original_label] = hardness_level - 1  # 从0开始
            
            # 应用映射
            remapped_labels = np.array([label_mapping[label] for label in labels])
            
            logger.info(f"标签重映射完成: {dict(zip(unique_labels, remapped_labels[::len(unique_labels)]))}")
            return remapped_labels
            
        except Exception as e:
            logger.error(f"标签重映射失败: {e}")
            return labels
    
    def create_hardness_grid_for_sample(self, file_path):
        """为单个样本创建硬度分数网格"""
        if self.coordinates is None:
            logger.error("坐标数据未加载")
            return None
        
        try:
            # 提取该样本的Paxini数据
            df = pd.read_csv(file_path, header=None, 
                           names=[f'col_{i}' for i in range(730)], 
                           low_memory=False)
            
            # 获取峰值点的Paxini数据
            peak_index = df[f'col_{CONFIG["FORCE_Z_INDEX"]}'].idxmin()
            paxini_values = []
            
            for contact_idx in range(CONFIG['PAXINI_NUM_POINTS']):
                paxini_col = f'col_{CONFIG["PAXINI_START_INDEX"] + contact_idx}'
                if paxini_col in df.columns:
                    value = pd.to_numeric(df.loc[peak_index, paxini_col], errors='coerce')
                    if not np.isnan(value):
                        paxini_values.append(value)
                    else:
                        paxini_values.append(0)
                else:
                    paxini_values.append(0)
            
            # 使用坐标数据进行网格插值
            points = self.coordinates[:, :2]  # 只使用XY坐标
            values = np.array(paxini_values)
            
            # 创建规则网格
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            grid_x, grid_y = np.mgrid[x_min:x_max:CONFIG['GRID_SHAPE'][1]*1j, 
                                      y_min:y_max:CONFIG['GRID_SHAPE'][0]*1j]
            
            # 插值到规则网格
            grid_scores = griddata(
                points, values, (grid_x, grid_y), 
                method='linear', fill_value=np.mean(values)
            )
            
            return grid_scores.T
            
        except Exception as e:
            logger.error(f"创建硬度网格失败: {e}")
            # 返回一个默认网格
            return np.full(CONFIG['GRID_SHAPE'], 128)  # 中性值
    
    def predict_single_file(self, file_path):
        """预测单个文件的硬度"""
        if self.cluster_model is None or self.scaler is None:
            logger.error("模型未训练")
            return None, None, None
            
        try:
            # 提取特征
            sample_features = self.extract_features_from_file(file_path)
            if not sample_features:
                return None, None, None
            
            # 构建特征向量
            feature_vector = [sample_features.get(key, 0) for key in self.feature_names]
            
            # 预测
            features_scaled = self.scaler.transform([feature_vector])
            label = self.cluster_model.predict(features_scaled)[0]
            
            # 生成网格
            grid_scores = self.create_hardness_grid_for_sample(file_path)
            
            return label, grid_scores, sample_features
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None, None, None
    
    def save_model(self, model_path):
        """保存模型"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'cluster_model': self.cluster_model,
                    'feature_names': self.feature_names,
                    'coordinates': self.coordinates
                }, f)
            logger.info(f"模型已保存到: {model_path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, model_path):
        """加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.cluster_model = model_data['cluster_model']
            self.feature_names = model_data['feature_names']
            self.coordinates = model_data['coordinates']
            logger.info(f"模型已从 {model_path} 加载")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def save_results(self, hardness_scores, grid_scores_dict, clustering_info):
        """保存结果"""
        try:
            # 保存样本硬度分数
            results_df = pd.DataFrame({
                'file_name': self.file_names,
                'hardness_level': hardness_scores + 1,  # 从1开始计数
                'original_cluster': clustering_info['original_labels']
            })
            results_path = os.path.join(CONFIG['OUTPUT_DIR'], 'hardness_assessment_results.csv')
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            
            # 保存每个样本的9×11网格
            for filename, grid_scores in grid_scores_dict.items():
                base_name = os.path.splitext(filename)[0]
                grid_path = os.path.join(CONFIG['OUTPUT_DIR'], f'{base_name}_hardness_grid_9x11.csv')
                grid_df = pd.DataFrame(grid_scores)
                grid_df.to_csv(grid_path, index=False, header=False)
            
            # 保存聚类信息
            info_path = os.path.join(CONFIG['OUTPUT_DIR'], 'clustering_info.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"聚类算法: {clustering_info['algorithm']}\n")
                f.write(f"最佳聚类数: {clustering_info['optimal_clusters']}\n")
                f.write(f"轮廓系数: {clustering_info['silhouette_score']:.4f}\n")
                f.write(f"特征数量: {len(self.feature_names)}\n")
                f.write(f"总样本数: {len(hardness_scores)}\n")
                f.write("特征列表:\n")
                for name in self.feature_names:
                    f.write(f"  - {name}\n")
                f.write("\n样本硬度分布:\n")
                for level in range(clustering_info['optimal_clusters']):
                    count = np.sum(hardness_scores == level)
                    f.write(f"  硬度等级 {level + 1}: {count} 个样本\n")
            
            logger.info(f"结果已保存到 {CONFIG['OUTPUT_DIR']}")
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False