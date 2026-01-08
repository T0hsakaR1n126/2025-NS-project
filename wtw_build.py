# network construction
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UndirectedWTWNetwork:
    year: int
    countries: List[str]
    adjacency_matrix: np.ndarray
    degrees: Dict[str, int]
    gdp_values: Dict[str, float]
    fitness_values: Dict[str, float]
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
        return self.degrees.get(country_code, 0)
    
    def are_connected(self, country1: str, country2: str) -> bool:
        if country1 not in self.countries or country2 not in self.countries:
            return False
        idx1 = self.countries.index(country1)
        idx2 = self.countries.index(country2)
        return self.adjacency_matrix[idx1, idx2] == 1
    
    def to_dataframe(self):
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
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.trade_data = None
        self.gdp_data = None
        self.networks = {}
        
        self._check_required_files()
    
    def _check_required_files(self):
        required_files = {
            'GDP data': os.path.join(self.raw_dir, 'world_bank_gdp.csv'),
            'Export data': os.path.join(self.raw_dir, 'comtrade_export_raw.csv'),
            'Import data': os.path.join(self.raw_dir, 'comtrade_import_raw.csv')
        }
        
        missing_files = []
        for file_type, file_path in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{file_type}: {file_path}")
        
        if missing_files:
            logger.warning(f"Missing files:\n" + "\n".join(missing_files))
            logger.warning("Ensure files are in correct location")
        else:
            print("All required files exist")
    
    def load_gdp_data(self):
        gdp_file = os.path.join(self.raw_dir, 'world_bank_gdp.csv')
        
        if not os.path.exists(gdp_file):
            alt_files = [
                os.path.join(self.data_dir, 'TotalGDP_2000~2020.csv'),
                os.path.join(self.processed_dir, 'TotalGDP_2000~2020.csv'),
                os.path.join(self.raw_dir, 'TotalGDP_2000~2020.csv')
            ]
            
            for alt_file in alt_files:
                if os.path.exists(alt_file):
                    gdp_file = alt_file
                    print(f"Using alternative GDP file: {gdp_file}")
                    break
        
        print(f"Loading GDP data: {gdp_file}")
        
        try:
            self.gdp_data = pd.read_csv(gdp_file)
            
            print(f"GDP data shape: {self.gdp_data.shape}")
            print(f"GDP columns: {self.gdp_data.columns.tolist()}")
            
            self._clean_gdp_data()
            
            print(f"GDP data loaded: {len(self.gdp_data)} records")
            print(f"Year range: {self.gdp_data['year'].min()} - {self.gdp_data['year'].max()}")
            print(f"Number of countries: {self.gdp_data['iso3c'].nunique()}")
            
            return self.gdp_data
            
        except Exception as e:
            logger.error(f"Failed to load GDP data: {e}")
            raise
    
    def _clean_gdp_data(self):
        column_mapping = {
            'iso3c': ['iso3c', 'country_code', 'iso', 'iso3'],
            'year': ['year', 'Year', 'period', 'Period'],
            'gdp_const_2015_usd': ['gdp_const_2015_usd', 'gdp', 'GDP', 'value', 'Value']
        }
        
        for target_col, possible_names in column_mapping.items():
            for name in possible_names:
                if name in self.gdp_data.columns and target_col not in self.gdp_data.columns:
                    self.gdp_data = self.gdp_data.rename(columns={name: target_col})
                    print(f"Renamed column: {name} -> {target_col}")
                    break
        
        required_cols = ['iso3c', 'year']
        missing_cols = [col for col in required_cols if col not in self.gdp_data.columns]
        
        if missing_cols:
            logger.error(f"GDP data missing required columns: {missing_cols}")
            logger.error(f"Available columns: {self.gdp_data.columns.tolist()}")
            raise ValueError(f"GDP data missing columns: {missing_cols}")
        
        if 'gdp_const_2015_usd' not in self.gdp_data.columns:
            gdp_candidates = ['gdp', 'GDP', 'value', 'Value', 'gdppc', 'GDPPC']
            for col in gdp_candidates:
                if col in self.gdp_data.columns:
                    self.gdp_data = self.gdp_data.rename(columns={col: 'gdp_const_2015_usd'})
                    print(f"Renamed GDP column: {col} -> gdp_const_2015_usd")
                    break
            
            if 'gdp_const_2015_usd' not in self.gdp_data.columns:
                logger.warning("No GDP column found, creating default values")
                self.gdp_data['gdp_const_2015_usd'] = 1.0
        
        self.gdp_data['iso3c'] = self.gdp_data['iso3c'].astype(str).str.strip().str.upper()
        self.gdp_data['year'] = pd.to_numeric(self.gdp_data['year'], errors='coerce')
        self.gdp_data['gdp_const_2015_usd'] = pd.to_numeric(
            self.gdp_data['gdp_const_2015_usd'], errors='coerce'
        )
        
        original_len = len(self.gdp_data)
        self.gdp_data = self.gdp_data.dropna(subset=['iso3c', 'year', 'gdp_const_2015_usd'])
        self.gdp_data = self.gdp_data[self.gdp_data['gdp_const_2015_usd'] > 0]
        
        removed_count = original_len - len(self.gdp_data)
        if removed_count > 0:
            print(f"Removed {removed_count} invalid GDP rows")
        
        self.gdp_data['year'] = self.gdp_data['year'].astype(int)
    
    def load_trade_data(self):
        export_file = os.path.join(self.raw_dir, 'comtrade_export_raw.csv')
        import_file = os.path.join(self.raw_dir, 'comtrade_import_raw.csv')
        
        try:
            export_data = pd.read_csv(export_file, low_memory=False)
            import_data = pd.read_csv(import_file, low_memory=False)
            self.trade_data = pd.concat([export_data, import_data], ignore_index=True)
            
            return self.trade_data
            
        except Exception as e:
            logger.error(f"Failed to load trade data: {e}")
            raise
    
    def process_trade_data(self, min_trade_value: float = 0):
        if self.trade_data is None:
            self.load_trade_data()
        
        reporter_col, partner_col, value_col, year_col = self._identify_columns()
        
        if not reporter_col or not partner_col:
            logger.error("Cannot identify country code columns")
            return {}
        
        processed_data = pd.DataFrame()
        processed_data['reporter'] = self.trade_data[reporter_col]
        processed_data['partner'] = self.trade_data[partner_col]
        
        if value_col:
            processed_data['trade_value'] = self.trade_data[value_col]
        else:
            processed_data['trade_value'] = 1
        
        if year_col:
            processed_data['year'] = self.trade_data[year_col]
        else:
            processed_data['year'] = 2000
        
        print(f"Processed data columns: {processed_data.columns.tolist()}")
        print(f"Processed data shape: {processed_data.shape}")
        
        processed_data = self._clean_trade_data(processed_data, min_trade_value)
        
        year_groups = {}
        for year, group in processed_data.groupby('year'):
            year_groups[year] = group
        
        processed_file = os.path.join(self.processed_dir, 'undirected_trade_data.csv')
        processed_data.to_csv(processed_file, index=False)
        print(f"Processed data saved: {processed_file}")
        
        for year, group in year_groups.items():
            year_file = os.path.join(self.processed_dir, f'trade_data_{year}.csv')
            group.to_csv(year_file, index=False)
        
        return year_groups
    
    def _identify_columns(self):
        columns = self.trade_data.columns.tolist()
        
        reporter_candidates = ['reporterISO', 'reporterCode', 'rtCode', 'rt3ISO', 'Reporter ISO Code', 'Reporter Code']
        reporter_col = None
        for col in reporter_candidates:
            if col in columns:
                reporter_col = col
                break
        
        partner_candidates = ['partnerISO', 'partnerCode', 'ptCode', 'pt3ISO', 'Partner ISO Code', 'Partner Code']
        partner_col = None
        for col in partner_candidates:
            if col in columns:
                partner_col = col
                break
        
        value_candidates = ['tradeValue', 'Trade Value', 'primaryValue', 'cifvalue', 'fobvalue', 'value']
        value_col = None
        for col in value_candidates:
            if col in columns:
                value_col = col
                break
        
        year_candidates = ['period', 'year', 'refYear', 'Year', 'periodDesc']
        year_col = None
        for col in year_candidates:
            if col in columns:
                year_col = col
                break
        
        return reporter_col, partner_col, value_col, year_col
    
    def _clean_trade_data(self, data: pd.DataFrame, min_trade_value: float) -> pd.DataFrame:
        original_len = len(data)
        
        data = data.dropna(subset=['reporter', 'partner'])
        
        data['reporter'] = data['reporter'].astype(str).str.strip().str.upper()
        data['partner'] = data['partner'].astype(str).str.strip().str.upper()
        
        invalid_codes = ['WLD', 'WORLD', 'W00', '0', '000', 'N/A', 'NA', '', 
                        'TOTAL', 'ALL', 'UNSPECIFIED']
        mask = (~data['reporter'].isin(invalid_codes)) & \
               (~data['partner'].isin(invalid_codes))
        data = data[mask]
        
        mask = data['reporter'] != data['partner']
        data = data[mask]
        
        if 'trade_value' in data.columns:
            data['trade_value'] = pd.to_numeric(data['trade_value'], errors='coerce')
            
            if min_trade_value > 0:
                data = data[data['trade_value'] >= min_trade_value]
            else:
                data = data[data['trade_value'] > 0]
        
        if 'year' in data.columns:
            data['year'] = pd.to_numeric(data['year'], errors='coerce').astype('Int64')
            data = data.dropna(subset=['year'])
            data['year'] = data['year'].astype(int)
        
        cleaned_len = len(data)
        print(f"Trade data cleaned: {original_len} → {cleaned_len} records")
        
        return data
    
    def build_undirected_networks(self, trade_year_groups: Dict[int, pd.DataFrame],
                                  min_countries: int = 50):
        if self.gdp_data is None:
            self.load_gdp_data()
        
        if not trade_year_groups:
            logger.error("No trade data")
            return {}
        
        print("Building undirected unweighted trade networks...")
        
        for year, trade_df in tqdm(trade_year_groups.items(), desc="Building Networks"):
            try:
                network = self._build_undirected_network(year, trade_df, min_countries)
                if network:
                    self.networks[year] = network
                    print(f"  {year}: {network.num_nodes} countries, {network.num_edges} edges, "
                              f"density={network.network_density:.4f}")
            except Exception as e:
                logger.error(f"Failed to build network for {year}: {e}")
        
        print(f"Network building complete: {len(self.networks)} years")
        return self.networks
    
    def _build_undirected_network(self, year: int, trade_df: pd.DataFrame, 
                                  min_countries: int) -> UndirectedWTWNetwork:
        logger.debug(f"Building undirected network for {year}...")
        
        year_gdp = self.gdp_data[self.gdp_data['year'] == year].copy()
        
        if year_gdp.empty:
            logger.warning(f"No GDP data for {year}")
            return None
        
        trade_countries = set(trade_df['reporter'].unique()) | set(trade_df['partner'].unique())
        gdp_countries = set(year_gdp['iso3c'].unique())
        
        common_countries = sorted(list(trade_countries.intersection(gdp_countries)))
        
        if len(common_countries) < min_countries:
            logger.warning(f"Too few common countries for {year}: {len(common_countries)} < {min_countries}")
            return None
        
        logger.debug(f"  {year}: {len(common_countries)} common countries")
        
        trade_pairs = set()
        
        for _, row in trade_df.iterrows():
            reporter = row['reporter']
            partner = row['partner']
            
            if reporter in common_countries and partner in common_countries:
                if reporter < partner:
                    edge = (reporter, partner)
                else:
                    edge = (partner, reporter)
                trade_pairs.add(edge)
        
        logger.debug(f"  {year}: {len(trade_pairs)} trade relationships")
        
        n = len(common_countries)
        country_index = {country: idx for idx, country in enumerate(common_countries)}
        adj_matrix = np.zeros((n, n), dtype=int)
        
        for country1, country2 in trade_pairs:
            i = country_index[country1]
            j = country_index[country2]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
        
        degrees = {}
        for country, idx in country_index.items():
            degrees[country] = int(np.sum(adj_matrix[idx]))
        
        gdp_values = {}
        fitness_values = {}
        
        gdp_dict = {row['iso3c']: row['gdp_const_2015_usd'] for _, row in year_gdp.iterrows()}
        
        total_gdp = sum(gdp_dict.get(country, 0) for country in common_countries)
        
        if total_gdp <= 0:
            logger.warning(f"Total GDP is 0 or negative for {year}")
            return None
        
        for country in common_countries:
            gdp = gdp_dict.get(country, 0)
            if gdp > 0:
                gdp_values[country] = gdp
                fitness_values[country] = gdp / total_gdp
            else:
                gdp_values[country] = 0
                fitness_values[country] = 0
        
        num_edges = int(np.sum(adj_matrix) / 2)
        
        network = UndirectedWTWNetwork(
            year=year,
            countries=common_countries,
            adjacency_matrix=adj_matrix,
            degrees=degrees,
            gdp_values=gdp_values,
            fitness_values=fitness_values,
            num_edges=num_edges
        )
        
        logger.debug(f"  {year} network built: {n} countries, {num_edges} edges")
        
        return network
    
    def save_networks(self, filename: str = 'undirected_wtw_networks.pkl'):
        if not self.networks:
            logger.warning("No network data to save")
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
        
        print(f"Network data saved: {file_path}")
        
        self._save_network_summary()
    
    def _save_network_summary(self):
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
        
        csv_path = os.path.join(self.processed_dir, 'undirected_network_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        
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
        
        print(f"Network summary saved: {csv_path}, {detail_path}")
    
    def load_networks(self, filename: str = 'undirected_wtw_networks.pkl'):
        file_path = os.path.join(self.processed_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.networks = saved_data['networks']
        print(f"Loaded {len(self.networks)} years of network data")
        
        return self.networks
    
    def get_network_statistics(self) -> pd.DataFrame:
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
    print("=" * 70)
    print("NetWork Construction")
    print("=" * 70)
    
    DATA_DIR = './data'
    MIN_COUNTRIES = 50
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Minimum countries: {MIN_COUNTRIES}")
    
    try:
        print("\n1. Initializing data processor...")
        processor = OfflineDataProcessor(data_dir=DATA_DIR)
        
        print("\n2. Loading GDP data...")
        gdp_data = processor.load_gdp_data()
        
        print("\n3. Loading and processing trade data...")
        year_groups = processor.process_trade_data(min_trade_value=0)
        
        if not year_groups:
            print("Warning: Failed to process trade data")
            return
        
        years = sorted(year_groups.keys())
        print(f"\nProcessed {len(years)} years of data: from {min(years)} to {max(years)}")
        
        print("\n4. Building undirected unweighted networks...")
        networks = processor.build_undirected_networks(year_groups, min_countries=MIN_COUNTRIES)
        
        if not networks:
            print("Warning: Failed to build networks")
            return
        
        print("\n5. Saving network data...")
        processor.save_networks()
        
        print("\n" + "=" * 60)
        print("Data processing complete!")
        print(f"Network data saved to: {processor.processed_dir}")
        
        print("\nNetwork summary:")
        print("-" * 40)
        
        years = sorted(networks.keys())
        for year in years:
            if year in networks:
                net = networks[year]
                print(f"  {year}: {net.num_nodes} countries, {net.num_edges} edges, "
                      f"density={net.network_density:.4f}")
        
        print(f"\nBuilt {len(networks)} years of network data")
        
        stats_df = processor.get_network_statistics()
        if not stats_df.empty:
            avg_nodes = stats_df['num_countries'].mean()
            avg_edges = stats_df['num_edges'].mean()
            avg_density = stats_df['density'].mean()
            min_nodes = stats_df['num_countries'].min()
            max_nodes = stats_df['num_countries'].max()
            
            print("\nNetwork statistics:")
            print(f"  Average countries per year: {avg_nodes:.0f}")
            print(f"  Country range: {min_nodes} to {max_nodes}")
            print(f"  Average edges per year: {avg_edges:.0f}")
            print(f"  Average density: {avg_density:.4f}")
            print(f"  Data period: {min(years)} to {max(years)}")
        
        return processor
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_offline_processing()