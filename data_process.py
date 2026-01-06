"""
data_process.py

贸易数据处理模块 - 离线版本
基于现有数据文件处理，不进行API下载
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UndirectedWTWNetwork:
    """无向无权世界贸易网络（文献格式）"""
    year: int
    countries: List[str]  # ISO3代码列表
    adjacency_matrix: np.ndarray  # 邻接矩阵（0/1）
    degrees: Dict[str, int]  # 国家度值
    gdp_values: Dict[str, float]  # 国家GDP（不变价2015美元）
    fitness_values: Dict[str, float]  # 归一化适应度值
    num_edges: int
    
    @property
    def num_nodes(self):
        return len(self.countries)
    
    @property
    def network_density(self):
        if self.num_nodes <= 1:
            return 0
        return (2 * self.num_edges) / (self.num_nodes * (self.num_nodes - 1))
    
    def get_country_degree(self, country_code: str) -> int:
        """获取国家度值"""
        return self.degrees.get(country_code, 0)
    
    def are_connected(self, country1: str, country2: str) -> bool:
        """检查两个国家是否相连"""
        if country1 not in self.countries or country2 not in self.countries:
            return False
        idx1 = self.countries.index(country1)
        idx2 = self.countries.index(country2)
        return self.adjacency_matrix[idx1, idx2] == 1
    
    def to_dataframe(self):
        """转换为DataFrame格式"""
        data = []
        for country in self.countries:
            data.append({
                'country': country,
                'year': self.year,
                'degree': self.degrees[country],
                'gdp': self.gdp_values.get(country, 0),
                'fitness': self.fitness_values.get(country, 0),
                'num_edges': self.num_edges,
                'density': self.network_density
            })
        return pd.DataFrame(data)


class OfflineDataProcessor:
    """离线数据处理器（使用已有文件）"""
    
    def __init__(self, data_dir: str = './data'):
        """
        初始化处理器
        
        参数：
        ----------
        data_dir : str
            数据目录路径
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # 检查目录是否存在
        if not os.path.exists(self.processed_dir):
            logger.error(f"处理目录不存在: {self.processed_dir}")
            raise FileNotFoundError(f"目录不存在: {self.processed_dir}")
        
        # 数据存储
        self.trade_data = None
        self.gdp_data = None
        self.networks = {}  # {year: UndirectedWTWNetwork}
        
        logger.info(f"离线数据处理器初始化完成，数据目录: {data_dir}")
        
        # 检查必要文件
        self._check_required_files()
    
    def _check_required_files(self):
        """检查必要的文件是否存在"""
        required_files = {
            'GDP数据': os.path.join(self.processed_dir, 'world_bank_gdp.csv'),
            '出口数据': os.path.join(self.processed_dir, 'comtrade_export_raw.csv'),
            '进口数据': os.path.join(self.processed_dir, 'comtrade_import_raw.csv')
        }
        
        missing_files = []
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")
        
        if missing_files:
            logger.warning(f"缺少以下文件:\n" + "\n".join(missing_files))
            logger.warning("请确保文件在正确位置")
        else:
            logger.info("所有必需文件都存在")
    
    def load_gdp_data(self):
        """加载GDP数据"""
        gdp_file = os.path.join(self.processed_dir, 'world_bank_gdp.csv')
        
        if not os.path.exists(gdp_file):
            # 尝试备选文件
            alt_files = [
                os.path.join(self.data_dir, 'TotalGDP_2000~2020.csv'),
                os.path.join(self.processed_dir, 'TotalGDP_2000~2020.csv')
            ]
            
            for alt_file in alt_files:
                if os.path.exists(alt_file):
                    gdp_file = alt_file
                    logger.info(f"使用备选GDP文件: {gdp_file}")
                    break
        
        logger.info(f"加载GDP数据: {gdp_file}")
        
        try:
            self.gdp_data = pd.read_csv(gdp_file)
            
            # 显示数据信息
            logger.info(f"GDP数据形状: {self.gdp_data.shape}")
            logger.info(f"GDP列名: {self.gdp_data.columns.tolist()}")
            
            # 数据清洗和标准化
            self._clean_gdp_data()
            
            logger.info(f"GDP数据加载完成: {len(self.gdp_data)}条记录")
            logger.info(f"年份范围: {self.gdp_data['year'].min()} - {self.gdp_data['year'].max()}")
            logger.info(f"国家数量: {self.gdp_data['iso3c'].nunique()}")
            
            return self.gdp_data
            
        except Exception as e:
            logger.error(f"加载GDP数据失败: {e}")
            raise
    
    def _clean_gdp_data(self):
        """清洗GDP数据"""
        # 重命名列（如果需要）
        column_mapping = {
            'iso3c': ['iso3c', 'country_code', 'iso', 'iso3'],
            'year': ['year', 'Year', 'period', 'Period'],
            'gdp_const_2015_usd': ['gdp_const_2015_usd', 'gdp', 'GDP', 'value', 'Value']
        }
        
        # 查找并重命名列
        for target_col, possible_names in column_mapping.items():
            for name in possible_names:
                if name in self.gdp_data.columns and target_col not in self.gdp_data.columns:
                    self.gdp_data = self.gdp_data.rename(columns={name: target_col})
                    logger.info(f"重命名列: {name} -> {target_col}")
                    break
        
        # 验证必要列
        required_cols = ['iso3c', 'year']
        missing_cols = [col for col in required_cols if col not in self.gdp_data.columns]
        
        if missing_cols:
            logger.error(f"GDP数据缺少必要列: {missing_cols}")
            logger.error(f"可用列: {self.gdp_data.columns.tolist()}")
            raise ValueError(f"GDP数据缺少列: {missing_cols}")
        
        # 确保有GDP值列
        if 'gdp_const_2015_usd' not in self.gdp_data.columns:
            # 查找可能的GDP列
            gdp_candidates = ['gdp', 'GDP', 'value', 'Value', 'gdppc', 'GDPPC']
            for col in gdp_candidates:
                if col in self.gdp_data.columns:
                    self.gdp_data = self.gdp_data.rename(columns={col: 'gdp_const_2015_usd'})
                    logger.info(f"重命名GDP列: {col} -> gdp_const_2015_usd")
                    break
            
            if 'gdp_const_2015_usd' not in self.gdp_data.columns:
                # 创建默认GDP列
                logger.warning("未找到GDP列，创建默认值")
                self.gdp_data['gdp_const_2015_usd'] = 1.0
        
        # 数据清洗
        self.gdp_data['iso3c'] = self.gdp_data['iso3c'].astype(str).str.strip().str.upper()
        self.gdp_data['year'] = pd.to_numeric(self.gdp_data['year'], errors='coerce')
        self.gdp_data['gdp_const_2015_usd'] = pd.to_numeric(
            self.gdp_data['gdp_const_2015_usd'], errors='coerce'
        )
        
        # 移除无效数据
        original_len = len(self.gdp_data)
        self.gdp_data = self.gdp_data.dropna(subset=['iso3c', 'year', 'gdp_const_2015_usd'])
        self.gdp_data = self.gdp_data[self.gdp_data['gdp_const_2015_usd'] > 0]
        
        removed_count = original_len - len(self.gdp_data)
        if removed_count > 0:
            logger.info(f"移除{removed_count}个无效GDP数据行")
        
        # 确保年份是整数
        self.gdp_data['year'] = self.gdp_data['year'].astype(int)
    
    def load_trade_data(self):
        """加载贸易数据"""
        export_file = os.path.join(self.processed_dir, 'comtrade_export_raw.csv')
        import_file = os.path.join(self.processed_dir, 'comtrade_import_raw.csv')
        
        logger.info("加载贸易数据...")
        
        try:
            # 加载出口数据
            export_data = pd.read_csv(export_file, low_memory=False)
            logger.info(f"出口数据形状: {export_data.shape}")
            
            # 加载进口数据
            import_data = pd.read_csv(import_file, low_memory=False)
            logger.info(f"进口数据形状: {import_data.shape}")
            
            # 合并数据
            self.trade_data = pd.concat([export_data, import_data], ignore_index=True)
            logger.info(f"合并后贸易数据形状: {self.trade_data.shape}")
            
            # 显示数据信息
            logger.info(f"贸易数据列名: {self.trade_data.columns.tolist()}")
            
            return self.trade_data
            
        except Exception as e:
            logger.error(f"加载贸易数据失败: {e}")
            raise
    
    def process_trade_data(self, min_trade_value: float = 0):
      """
      处理贸易数据为无向无权网络格式
      """
      if self.trade_data is None:
          self.load_trade_data()
      
      logger.info("处理贸易数据为无向无权网络格式...")
      
      # 确定列名
      reporter_col, partner_col, value_col, year_col = self._identify_columns()
      
      if not reporter_col or not partner_col:
          logger.error("无法确定国家代码列")
          return {}
      
      logger.info(f"使用列名: reporter={reporter_col}, partner={partner_col}, "
                f"value={value_col}, year={year_col}")
      
      # 创建新DataFrame时直接使用正确的列名
      processed_data = pd.DataFrame()
      processed_data['reporter'] = self.trade_data[reporter_col]
      processed_data['partner'] = self.trade_data[partner_col]
      
      # 添加值和年份列
      if value_col:
          processed_data['trade_value'] = self.trade_data[value_col]
      else:
          processed_data['trade_value'] = 1  # 默认值
      
      if year_col:
          processed_data['year'] = self.trade_data[year_col]
      else:
          processed_data['year'] = 2000  # 默认年份
      
      logger.info(f"处理后数据列名: {processed_data.columns.tolist()}")
      logger.info(f"处理前数据形状: {processed_data.shape}")
      
      # 数据清洗
      processed_data = self._clean_trade_data(processed_data, min_trade_value)
      
      # 按年份分组
      year_groups = {}
      for year, group in processed_data.groupby('year'):
          year_groups[year] = group
      
      # 保存处理后的数据
      processed_file = os.path.join(self.processed_dir, 'undirected_trade_data.csv')
      processed_data.to_csv(processed_file, index=False)
      logger.info(f"处理后数据已保存: {processed_file}")
      
      # 保存按年份分组的数据
      for year, group in year_groups.items():
          year_file = os.path.join(self.processed_dir, f'trade_data_{year}.csv')
          group.to_csv(year_file, index=False)
      
      logger.info(f"按年份分组数据已保存，共{len(year_groups)}个年份")
      
      return year_groups
    
    def _identify_columns(self):
        """识别贸易数据中的列名"""
        columns = self.trade_data.columns.tolist()
        
        # 查找reporter列
        reporter_candidates = ['reporterISO', 'reporterCode', 'rtCode', 'rt3ISO', 
                              'Reporter ISO Code', 'Reporter Code']
        reporter_col = None
        for col in reporter_candidates:
            if col in columns:
                reporter_col = col
                break
        
        # 查找partner列
        partner_candidates = ['partnerISO', 'partnerCode', 'ptCode', 'pt3ISO',
                             'Partner ISO Code', 'Partner Code']
        partner_col = None
        for col in partner_candidates:
            if col in columns:
                partner_col = col
                break
        
        # 查找value列
        value_candidates = ['tradeValue', 'Trade Value', 'primaryValue', 
                           'cifvalue', 'fobvalue', 'value']
        value_col = None
        for col in value_candidates:
            if col in columns:
                value_col = col
                break
        
        # 查找year列
        year_candidates = ['period', 'year', 'refYear', 'Year', 'periodDesc']
        year_col = None
        for col in year_candidates:
            if col in columns:
                year_col = col
                break
        
        return reporter_col, partner_col, value_col, year_col
    
    def _clean_trade_data(self, data: pd.DataFrame, min_trade_value: float) -> pd.DataFrame:
        """清洗贸易数据"""
        original_len = len(data)
        
        # 移除缺失值
        data = data.dropna(subset=['reporter', 'partner'])
        
        # 标准化国家代码
        data['reporter'] = data['reporter'].astype(str).str.strip().str.upper()
        data['partner'] = data['partner'].astype(str).str.strip().str.upper()
        
        # 移除无效国家代码
        invalid_codes = ['WLD', 'WORLD', 'W00', '0', '000', 'N/A', 'NA', '', 
                        'TOTAL', 'ALL', 'UNSPECIFIED']
        mask = (~data['reporter'].isin(invalid_codes)) & \
               (~data['partner'].isin(invalid_codes))
        data = data[mask]
        
        # 移除自环
        mask = data['reporter'] != data['partner']
        data = data[mask]
        
        # 处理贸易值
        if 'trade_value' in data.columns:
            data['trade_value'] = pd.to_numeric(data['trade_value'], errors='coerce')
            
            # 移除贸易额小于阈值的记录
            if min_trade_value > 0:
                data = data[data['trade_value'] >= min_trade_value]
            else:
                # 移除0或负值
                data = data[data['trade_value'] > 0]
        
        # 处理年份
        if 'year' in data.columns:
            data['year'] = pd.to_numeric(data['year'], errors='coerce').astype('Int64')
            data = data.dropna(subset=['year'])
            data['year'] = data['year'].astype(int)
        
        cleaned_len = len(data)
        logger.info(f"贸易数据清洗完成: 原始{original_len}条 → 清理后{cleaned_len}条")
        
        return data
    
    def build_undirected_networks(self, trade_year_groups: Dict[int, pd.DataFrame],
                                  min_countries: int = 50):
        """
        构建无向无权贸易网络
        
        参数：
        ----------
        trade_year_groups : Dict[int, pd.DataFrame]
            按年份分组的贸易数据
        min_countries : int
            最小国家数要求
        
        返回：
        ----------
        Dict[int, UndirectedWTWNetwork]
            构建的网络
        """
        if self.gdp_data is None:
            self.load_gdp_data()
        
        if not trade_year_groups:
            logger.error("没有贸易数据")
            return {}
        
        logger.info("开始构建无向无权贸易网络...")
        
        for year, trade_df in tqdm(trade_year_groups.items(), desc="构建网络"):
            try:
                network = self._build_undirected_network(year, trade_df, min_countries)
                if network:
                    self.networks[year] = network
                    logger.info(f"  {year}年: {network.num_nodes}国, {network.num_edges}边, "
                              f"密度={network.network_density:.4f}")
            except Exception as e:
                logger.error(f"构建{year}年网络失败: {e}")
        
        logger.info(f"网络构建完成: 共{len(self.networks)}个年份的网络")
        return self.networks
    
    def _build_undirected_network(self, year: int, trade_df: pd.DataFrame, 
                                  min_countries: int) -> UndirectedWTWNetwork:
        """
        构建单一年份的无向无权网络
        """
        logger.debug(f"构建{year}年无向无权网络...")
        
        # 1. 获取当年的GDP数据
        year_gdp = self.gdp_data[self.gdp_data['year'] == year].copy()
        
        if year_gdp.empty:
            logger.warning(f"{year}年无GDP数据")
            return None
        
        # 2. 从贸易数据中提取国家集合
        trade_countries = set(trade_df['reporter'].unique()) | set(trade_df['partner'].unique())
        gdp_countries = set(year_gdp['iso3c'].unique())
        
        # 3. 获取共同国家集合
        common_countries = sorted(list(trade_countries.intersection(gdp_countries)))
        
        if len(common_countries) < min_countries:
            logger.warning(f"{year}年共同国家太少: {len(common_countries)} < {min_countries}")
            return None
        
        logger.debug(f"  {year}年: {len(common_countries)}个共同国家")
        
        # 4. 构建贸易关系集合（无向）
        trade_pairs = set()
        
        for _, row in trade_df.iterrows():
            reporter = row['reporter']
            partner = row['partner']
            
            # 只考虑共同国家
            if reporter in common_countries and partner in common_countries:
                # 创建无向边（按字母顺序排序）
                if reporter < partner:
                    edge = (reporter, partner)
                else:
                    edge = (partner, reporter)
                trade_pairs.add(edge)
        
        logger.debug(f"  {year}年: {len(trade_pairs)}个贸易关系")
        
        # 5. 创建邻接矩阵
        n = len(common_countries)
        country_index = {country: idx for idx, country in enumerate(common_countries)}
        adj_matrix = np.zeros((n, n), dtype=int)
        
        for country1, country2 in trade_pairs:
            i = country_index[country1]
            j = country_index[country2]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        # 6. 计算度值
        degrees = {}
        for country, idx in country_index.items():
            degrees[country] = int(np.sum(adj_matrix[idx]))
        
        # 7. 获取GDP值和适应度值
        gdp_values = {}
        fitness_values = {}
        
        # 创建GDP字典
        gdp_dict = {row['iso3c']: row['gdp_const_2015_usd'] for _, row in year_gdp.iterrows()}
        
        # 计算总GDP
        total_gdp = sum(gdp_dict.get(country, 0) for country in common_countries)
        
        if total_gdp <= 0:
            logger.warning(f"{year}年总GDP为0或负值")
            return None
        
        for country in common_countries:
            gdp = gdp_dict.get(country, 0)
            if gdp > 0:
                gdp_values[country] = gdp
                fitness_values[country] = gdp / total_gdp
            else:
                gdp_values[country] = 0
                fitness_values[country] = 0
        
        # 8. 计算边数
        num_edges = int(np.sum(adj_matrix) / 2)
        
        # 9. 创建网络对象
        network = UndirectedWTWNetwork(
            year=year,
            countries=common_countries,
            adjacency_matrix=adj_matrix,
            degrees=degrees,
            gdp_values=gdp_values,
            fitness_values=fitness_values,
            num_edges=num_edges
        )
        
        logger.debug(f"  {year}年网络构建完成: {n}国, {num_edges}边")
        
        return network
    
    def save_networks(self, filename: str = 'undirected_wtw_networks.pkl'):
        """保存网络数据"""
        if not self.networks:
            logger.warning("没有网络数据可保存")
            return
        
        file_path = os.path.join(self.processed_dir, filename)
        
        save_data = {
            'networks': self.networks,
            'gdp_data_shape': self.gdp_data.shape if self.gdp_data is not None else None,
            'num_years': len(self.networks),
            'years': list(self.networks.keys())
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"网络数据已保存: {file_path}")
        
        # 同时保存文本格式的摘要
        self._save_network_summary()
    
    def _save_network_summary(self):
        """保存网络摘要"""
        summary_data = []
        
        for year, network in self.networks.items():
            degrees = list(network.degrees.values())
            gdp_values = [v for v in network.gdp_values.values() if v > 0]
            
            summary_data.append({
                'year': year,
                'num_countries': network.num_nodes,
                'num_edges': network.num_edges,
                'network_density': network.network_density,
                'mean_degree': np.mean(degrees) if degrees else 0,
                'std_degree': np.std(degrees) if degrees else 0,
                'max_degree': np.max(degrees) if degrees else 0,
                'min_degree': np.min(degrees) if degrees else 0,
                'total_gdp': sum(gdp_values) if gdp_values else 0,
                'mean_gdp': np.mean(gdp_values) if gdp_values else 0,
                'mean_fitness': np.mean(list(network.fitness_values.values())) if network.fitness_values else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存CSV格式
        csv_path = os.path.join(self.processed_dir, 'undirected_network_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        
        # 保存详细的网络统计
        detail_path = os.path.join(self.processed_dir, 'network_statistics.txt')
        with open(detail_path, 'w') as f:
            f.write("World Trade Web Statistics (Undirected, Unweighted)\n")
            f.write("=" * 60 + "\n\n")
            
            for year, network in self.networks.items():
                f.write(f"Year {year}:\n")
                f.write(f"  Countries: {network.num_nodes}\n")
                f.write(f"  Edges: {network.num_edges}\n")
                f.write(f"  Density: {network.network_density:.4f}\n")
                f.write(f"  Mean Degree: {np.mean(list(network.degrees.values())):.2f}\n")
                f.write(f"  Degree Range: [{np.min(list(network.degrees.values()))}, "
                       f"{np.max(list(network.degrees.values()))}]\n")
                f.write(f"  Total GDP: {sum(network.gdp_values.values()):.2e}\n")
                f.write(f"  Mean Fitness: {np.mean(list(network.fitness_values.values())):.6f}\n\n")
        
        logger.info(f"网络摘要已保存: {csv_path}, {detail_path}")
    
    def load_networks(self, filename: str = 'undirected_wtw_networks.pkl'):
        """加载已保存的网络数据"""
        file_path = os.path.join(self.processed_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.networks = saved_data['networks']
        logger.info(f"已加载{len(self.networks)}个年份的网络数据")
        
        return self.networks
    
    def get_network_statistics(self) -> pd.DataFrame:
        """获取网络统计信息"""
        if not self.networks:
            return pd.DataFrame()
        
        stats = []
        for year, network in self.networks.items():
            stats.append({
                'year': year,
                'num_countries': network.num_nodes,
                'num_edges': network.num_edges,
                'density': network.network_density,
                'mean_degree': np.mean(list(network.degrees.values())),
                'std_degree': np.std(list(network.degrees.values())),
                'min_degree': np.min(list(network.degrees.values())),
                'max_degree': np.max(list(network.degrees.values())),
                'total_gdp': sum(network.gdp_values.values()),
                'mean_gdp': np.mean([v for v in network.gdp_values.values() if v > 0])
            })
        
        return pd.DataFrame(stats)


def main_offline_processing():
    """主函数：执行离线数据处理"""
    print("=" * 60)
    print("世界贸易数据处理（离线版本）")
    print("=" * 60)
    
    # 配置参数 - 根据你的文件夹结构
    DATA_DIR = './data'  # 当前目录，因为你的processed文件夹在项目根目录
    MIN_COUNTRIES = 50
    
    print(f"数据目录: {DATA_DIR}")
    print(f"最小国家数: {MIN_COUNTRIES}")
    
    try:
        # 1. 初始化处理器
        print("\n1. 初始化数据处理器...")
        processor = OfflineDataProcessor(data_dir=DATA_DIR)
        
        # 2. 加载GDP数据
        print("\n2. 加载GDP数据...")
        gdp_data = processor.load_gdp_data()
        print(f"  已加载GDP数据: {len(gdp_data)}条记录")
        
        # 3. 加载和处理贸易数据
        print("\n3. 加载和处理贸易数据...")
        trade_data = processor.load_trade_data()
        print(f"  已加载贸易数据: {len(trade_data)}条记录")
        
        year_groups = processor.process_trade_data(min_trade_value=0)
        print(f"  处理为{len(year_groups)}个年份的数据组")
        
        if not year_groups:
            print("警告: 未能处理贸易数据")
            return
        
        # 显示可用的年份
        years = sorted(year_groups.keys())
        print(f"  可用年份: {years[:5]}... ({len(years)}年)")
        
        # 4. 构建无向无权网络
        print("\n4. 构建无向无权贸易网络...")
        networks = processor.build_undirected_networks(year_groups, min_countries=MIN_COUNTRIES)
        
        if not networks:
            print("警告: 未能构建网络")
            return
        
        # 5. 保存网络数据
        print("\n5. 保存网络数据...")
        processor.save_networks()
        
        print("\n" + "=" * 60)
        print("数据处理完成！")
        print(f"网络数据保存到: {processor.processed_dir}")
        
        # 显示摘要
        print("\n网络构建摘要:")
        print("-" * 40)
        
        years = sorted(networks.keys())
        for year in years:
            if year in networks:
                net = networks[year]
                print(f"  {year}年: {net.num_nodes}国, {net.num_edges}边, "
                      f"密度={net.network_density:.4f}")
        
        print(f"\n共构建 {len(networks)} 个年份的网络数据")
        
        # 保存详细的统计数据
        stats_df = processor.get_network_statistics()
        if not stats_df.empty:
            stats_file = os.path.join(processor.processed_dir, 'network_statistics_detailed.csv')
            stats_df.to_csv(stats_file, index=False)
            print(f"详细统计数据已保存: {stats_file}")
        
        return processor
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_offline_processing()