import numpy as np
import pandas as pd
from scipy import optimize, stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from nw_build import UndirectedWTWNetwork, OfflineDataProcessor
    
    TradeDataProcessor = OfflineDataProcessor
    
    print("Successfully imported wtw_build module")
    
except ImportError as e:
    logger.error(f"Failed to import wtw_build module: {e}")
    print(f"Failed to import wtw_build module: {e}")
    print("Make sure wtw_build.py is in the same directory")
    print("and contains UndirectedWTWNetwork and OfflineDataProcessor classes")
    raise

plt.style.use('seaborn-v0_8-darkgrid')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 100

@dataclass
class FitnessModelResult:
    year: int
    delta: float
    fitness_values: Dict[str, float]
    predicted_degrees: Dict[str, float]
    actual_degrees: Dict[str, int]
    countries: List[str]
    
    mae: float
    rmse: float
    r2: float
    
    num_nodes: int
    num_edges: int
    density: float
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        data = []
        for country in self.countries:
            data.append({
                'country': country,
                'fitness': self.fitness_values.get(country, 0),
                'degree_actual': self.actual_degrees.get(country, 0),
                'degree_predicted': self.predicted_degrees.get(country, 0),
                'degree_error': self.predicted_degrees.get(country, 0) - self.actual_degrees.get(country, 0),
                'relative_error': (self.predicted_degrees.get(country, 0) - self.actual_degrees.get(country, 0)) 
                                / max(1, self.actual_degrees.get(country, 0))
            })
        return pd.DataFrame(data)
    
    def get_summary_stats(self) -> Dict:
        actual_degrees = list(self.actual_degrees.values())
        predicted_degrees = list(self.predicted_degrees.values())
        
        return {
            'year': self.year,
            'delta': self.delta,
            'r2': self.r2,
            'mae': self.mae,
            'rmse': self.rmse,
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'density': self.density,
            'mean_degree_actual': np.mean(actual_degrees),
            'mean_degree_predicted': np.mean(predicted_degrees),
            'std_degree_actual': np.std(actual_degrees),
            'std_degree_predicted': np.std(predicted_degrees)
        }


class TopologyAnalyzer:
    def __init__(self, network: UndirectedWTWNetwork):
        self.network = network
        self.results = {}
    
    def analyze_all_properties(self) -> Dict:
        print(f"Analyzing topology for {self.network.year}")
        
        self.results['degree_distribution'] = self._analyze_degree_distribution()
        self.results['degree_correlations'] = self._analyze_degree_correlations()
        self.results['clustering'] = self._analyze_clustering()
        
        return self.results
    
    def _analyze_degree_distribution(self) -> Dict:
        degrees = list(self.network.degrees.values())
        
        result = {
            'degrees': degrees,
            'mean': float(np.mean(degrees)),
            'std': float(np.std(degrees)),
            'min': int(np.min(degrees)),
            'max': int(np.max(degrees)),
            'median': float(np.median(degrees)),
            'skewness': float(stats.skew(degrees)),
            'kurtosis': float(stats.kurtosis(degrees))
        }
        
        max_degree = int(np.max(degrees))
        degree_counts = np.bincount(degrees, minlength=max_degree+1)
        cumulative_dist = 1 - np.cumsum(degree_counts) / len(degrees)
        
        result['cumulative_distribution'] = {
            'k_values': np.arange(len(cumulative_dist)),
            'P_gt_k': cumulative_dist
        }
        
        try:
            k_vals = np.arange(1, len(cumulative_dist))
            p_vals = cumulative_dist[1:]
            
            mask = (k_vals > 0) & (p_vals > 0)
            if np.sum(mask) >= 5:
                log_k = np.log(k_vals[mask])
                log_p = np.log(p_vals[mask])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_p)
                
                result['power_law_fit'] = {
                    'exponent': -slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
        except Exception as e:
            logger.debug(f"Power law fit failed: {e}")
        
        return result
    
    def _analyze_degree_correlations(self) -> Dict:
        adj_matrix = self.network.adjacency_matrix
        countries = self.network.countries
        degrees = self.network.degrees
        
        knn_individual = {}
        
        for idx, country in enumerate(countries):
            ki = degrees[country]
            if ki > 0:
                neighbors = np.where(adj_matrix[idx] > 0)[0]
                neighbor_degrees = [degrees[countries[n]] for n in neighbors]
                knn_individual[country] = np.mean(neighbor_degrees)
            else:
                knn_individual[country] = 0
        
        unique_degrees = sorted(set(degrees.values()))
        knn_by_degree = {}
        degree_counts = {}
        
        for k in unique_degrees:
            countries_with_k = [c for c in countries if degrees[c] == k]
            if countries_with_k:
                knn_values = [knn_individual[c] for c in countries_with_k]
                knn_by_degree[k] = float(np.mean(knn_values))
                degree_counts[k] = len(countries_with_k)
        
        result = {
            'knn_individual': knn_individual,
            'knn_by_degree': knn_by_degree,
            'degree_counts': degree_counts
        }
        
        if len(knn_by_degree) >= 2:
            k_values = np.array(list(knn_by_degree.keys()))
            knn_values = np.array(list(knn_by_degree.values()))
            
            correlation = np.corrcoef(k_values, knn_values)[0, 1]
            result['assortativity'] = float(correlation)
            result['disassortative'] = correlation < 0
            
            if np.all(k_values > 0) and np.all(knn_values > 0):
                try:
                    log_k = np.log(k_values)
                    log_knn = np.log(knn_values)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_knn)
                    result['power_law_exponent'] = float(-slope)
                    result['power_law_r_squared'] = float(r_value**2)
                except:
                    pass
        
        return result
    
    def _analyze_clustering(self) -> Dict:
        adj_matrix = self.network.adjacency_matrix
        countries = self.network.countries
        degrees = self.network.degrees
        
        clustering_individual = {}
        
        for idx, country in enumerate(countries):
            ki = degrees[country]
            if ki < 2:
                clustering_individual[country] = 0.0
            else:
                neighbors = np.where(adj_matrix[idx] > 0)[0]
                num_neighbors = len(neighbors)
                
                triangles = 0
                for i in range(num_neighbors):
                    for j in range(i+1, num_neighbors):
                        if adj_matrix[neighbors[i], neighbors[j]] > 0:
                            triangles += 1
                
                clustering_individual[country] = 2.0 * triangles / (ki * (ki - 1))
        
        unique_degrees = sorted(set(degrees.values()))
        clustering_by_degree = {}
        
        for k in unique_degrees:
            if k >= 2:
                countries_with_k = [c for c in countries if degrees[c] == k]
                if countries_with_k:
                    clustering_values = [clustering_individual[c] for c in countries_with_k]
                    clustering_by_degree[k] = float(np.mean(clustering_values))
        
        global_clustering = float(np.mean(list(clustering_individual.values())))
        
        result = {
            'individual': clustering_individual,
            'by_degree': clustering_by_degree,
            'global': global_clustering
        }
        
        if len(clustering_by_degree) >= 2:
            k_values = np.array(list(clustering_by_degree.keys()))
            c_values = np.array(list(clustering_by_degree.values()))
            
            correlation = np.corrcoef(k_values, c_values)[0, 1]
            result['hierarchy_correlation'] = float(correlation)
            result['hierarchical'] = correlation < 0
        
        return result


class FitnessModelAnalyzer:
    def __init__(self, networks: Dict[int, UndirectedWTWNetwork], results_dir: str = './results'):
        self.networks = networks
        self.results_dir = results_dir
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        
        self.model_results: Dict[int, FitnessModelResult] = {}
        self.topology_results: Dict[int, Dict] = {}
        
        print(f"Fitness model analyzer initialized with {len(networks)} years")
    
    def fit_fitness_model(self, year: int, verbose: bool = False) -> Optional[FitnessModelResult]:
        if year not in self.networks:
            logger.error(f"No network data for {year}")
            return None
        
        network = self.networks[year]
        print(f"Fitting fitness model for {year}")
        
        fitness = network.fitness_values
        countries = network.countries
        
        valid_countries = [c for c in countries if fitness.get(c, 0) > 0]
        if len(valid_countries) < 10:
            logger.warning(f"Too few valid countries for {year}: {len(valid_countries)}")
            return None
        
        x_values = np.array([fitness[c] for c in valid_countries])
        actual_degrees = network.degrees
        k_actual = np.array([actual_degrees[c] for c in valid_countries])
        L_actual = network.num_edges
        
        def connection_probability(xi: float, xj: float, delta: float) -> float:
            return (delta * xi * xj) / (1 + delta * xi * xj)
        
        def objective(delta: float) -> float:
            n = len(x_values)
            L_pred = 0.0
            
            for i in range(n):
                for j in range(i+1, n):
                    L_pred += connection_probability(x_values[i], x_values[j], delta)
            
            return L_pred - L_actual
        
        try:
            delta_min, delta_max = 1e-10, 1e10
            
            f_min = objective(delta_min)
            f_max = objective(delta_max)
            
            if f_min * f_max > 0:
                logger.debug(f"{year}: Bisection boundaries same sign, using minimization")
                result = optimize.minimize_scalar(
                    lambda d: abs(objective(d)),
                    bounds=(delta_min, delta_max),
                    method='bounded',
                    options={'xatol': 1e-12}
                )
                delta_opt = result.x
            else:
                delta_opt = optimize.brentq(
                    objective,
                    delta_min,
                    delta_max,
                    xtol=1e-12,
                    maxiter=100
                )
            
            if verbose:
                print(f"{year}: δ = {delta_opt:.6e}")
                print(f"  Actual edges: {L_actual}")
                print(f"  Predicted edges: {objective(delta_opt) + L_actual:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to solve δ for {year}: {e}")
            return None
        
        n = len(valid_countries)
        k_pred_dict = {}
        k_pred_array = np.zeros(n)
        
        for idx, country in enumerate(valid_countries):
            xi = x_values[idx]
            k_pred = 0.0
            
            for j in range(n):
                if j != idx:
                    xj = x_values[j]
                    k_pred += connection_probability(xi, xj, delta_opt)
            
            k_pred_dict[country] = k_pred
            k_pred_array[idx] = k_pred
        
        mae = float(np.mean(np.abs(k_pred_array - k_actual)))
        rmse = float(np.sqrt(np.mean((k_pred_array - k_actual) ** 2)))
        
        ss_res = np.sum((k_actual - k_pred_array) ** 2)
        ss_tot = np.sum((k_actual - np.mean(k_actual)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
        
        if verbose:
            print(f"{year} model evaluation:")
            print(f"  MAE = {mae:.2f}")
            print(f"  RMSE = {rmse:.2f}")
            print(f"  R² = {r2:.4f}")
        
        result = FitnessModelResult(
            year=year,
            delta=delta_opt,
            fitness_values=fitness,
            predicted_degrees=k_pred_dict,
            actual_degrees=actual_degrees,
            countries=valid_countries,
            mae=mae,
            rmse=rmse,
            r2=r2,
            num_nodes=network.num_nodes,
            num_edges=network.num_edges,
            density=network.network_density
        )
        
        self.model_results[year] = result
        return result
    
    def analyze_topology(self, year: int) -> Optional[Dict]:
        if year not in self.networks:
            return None
        
        network = self.networks[year]
        analyzer = TopologyAnalyzer(network)
        results = analyzer.analyze_all_properties()
        
        self.topology_results[year] = results
        return results
    
    def plot_comparison_figure(self, year: int, save: bool = True) -> Optional[plt.Figure]:
        if year not in self.model_results or year not in self.topology_results:
            logger.error(f"Incomplete analysis results for {year}")
            return None
        
        model = self.model_results[year]
        topology = self.topology_results[year]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'World Trade Web Analysis - {year}', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        self._plot_degree_vs_fitness(ax1, model, topology)
        
        ax2 = axes[0, 1]
        self._plot_cumulative_degree_distribution(ax2, model, topology)
        
        ax3 = axes[1, 0]
        self._plot_average_nearest_neighbor_degree(ax3, model, topology)
        
        ax4 = axes[1, 1]
        self._plot_clustering_coefficient(ax4, model, topology)
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.results_dir, 'figures', f'wtw_analysis_{year}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_degree_vs_fitness(self, ax, model, topology):
        countries = model.countries
        fitness_vals = [model.fitness_values[c] for c in countries]
        k_actual = [model.actual_degrees[c] for c in countries]
        k_pred = [model.predicted_degrees[c] for c in countries]
        
        mask = np.array(fitness_vals) > 0
        if np.sum(mask) > 10:
            log_fitness = np.log10(np.array(fitness_vals)[mask])
            bins = np.linspace(log_fitness.min(), log_fitness.max(), 15)
            bin_centers = []
            bin_k_actual = []
            bin_k_pred = []
            
            for i in range(len(bins)-1):
                idx_mask = (log_fitness >= bins[i]) & (log_fitness < bins[i+1])
                if np.sum(idx_mask) > 0:
                    bin_center = 10 ** ((bins[i] + bins[i+1]) / 2)
                    actual_vals = [k_actual[j] for j, m in enumerate(mask) if m and idx_mask[j-np.sum(~mask)]]
                    pred_vals = [k_pred[j] for j, m in enumerate(mask) if m and idx_mask[j-np.sum(~mask)]]
                    
                    if actual_vals and pred_vals:
                        bin_centers.append(bin_center)
                        bin_k_actual.append(np.mean(actual_vals))
                        bin_k_pred.append(np.mean(pred_vals))
            
            ax.scatter(fitness_vals, k_actual, alpha=0.6, s=20, color='blue', label='Actual')
            
            if bin_centers:
                ax.plot(bin_centers, bin_k_actual, 'o-', color='red', linewidth=2, 
                       markersize=8, label='Actual (binned)')
                ax.plot(bin_centers, bin_k_pred, 's--', color='green', linewidth=2, 
                       markersize=6, label='Model prediction')
        
        ax.set_xscale('log')
        ax.set_xlabel('Fitness x (normalized GDP)', fontsize=12)
        ax.set_ylabel('Degree k', fontsize=12)
        ax.set_title(f'Degree vs Fitness (R² = {model.r2:.3f})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_degree_distribution(self, ax, model, topology):
        if 'degree_distribution' in topology and 'cumulative_distribution' in topology['degree_distribution']:
            cum_data = topology['degree_distribution']['cumulative_distribution']
            k_vals = cum_data['k_values']
            P_gt_k = cum_data['P_gt_k']
            
            mask = P_gt_k > 0
            if np.sum(mask) > 5:
                ax.plot(k_vals[mask], P_gt_k[mask], 'o-', linewidth=2, markersize=4, 
                       label='Empirical distribution')
                
                if 'power_law_fit' in topology['degree_distribution']:
                    fit_data = topology['degree_distribution']['power_law_fit']
                    exponent = fit_data['exponent']
                    
                    fit_k = k_vals[(k_vals > 0) & (k_vals < np.percentile(k_vals[mask], 95))]
                    if len(fit_k) > 0:
                        power_law = np.exp(fit_data['intercept']) * fit_k ** (-exponent)
                        ax.plot(fit_k, power_law, '--', color='red', linewidth=2,
                               label=f'Power law (γ={exponent:.2f})')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree k', fontsize=12)
        ax.set_ylabel('P(>k)', fontsize=12)
        ax.set_title('Cumulative Degree Distribution', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_average_nearest_neighbor_degree(self, ax, model, topology):
        if 'degree_correlations' in topology and 'knn_by_degree' in topology['degree_correlations']:
            knn_data = topology['degree_correlations']['knn_by_degree']
            if knn_data:
                k_vals = np.array(list(knn_data.keys()))
                knn_vals = np.array(list(knn_data.values()))
                
                ax.scatter(k_vals, knn_vals, alpha=0.7, s=50, color='blue')
                
                assort = topology['degree_correlations'].get('assortativity', 0)
                disassort = topology['degree_correlations'].get('disassortative', False)
                assort_text = 'Disassortative' if disassort else 'Assortative'
                ax.text(0.05, 0.95, assort_text, transform=ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if 'power_law_exponent' in topology['degree_correlations']:
                    exponent = topology['degree_correlations']['power_law_exponent']
                    fit_k = k_vals[(k_vals > 0) & (knn_vals > 0)]
                    if len(fit_k) > 2:
                        x_fit = np.linspace(fit_k.min(), fit_k.max(), 50)
                        y_fit = np.mean(knn_vals) * (x_fit / np.mean(fit_k)) ** (-exponent)
                        ax.plot(x_fit, y_fit, 'r--', linewidth=2,
                               label=f'K_nn(k) ∝ k^{{{-exponent:.2f}}}')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree k', fontsize=12)
        ax.set_ylabel(r'$K^{nn}(k)$', fontsize=12)
        ax.set_title('Average Nearest Neighbor Degree', fontsize=13)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_clustering_coefficient(self, ax, model, topology):
        if 'clustering' in topology and 'by_degree' in topology['clustering']:
            clust_data = topology['clustering']['by_degree']
            if clust_data:
                k_vals = np.array(list(clust_data.keys()))
                c_vals = np.array(list(clust_data.values()))
                
                ax.scatter(k_vals, c_vals, alpha=0.7, s=50, color='purple')
                
                is_hierarchical = topology['clustering'].get('hierarchical', False)
                hier_text = 'Hierarchical' if is_hierarchical else 'Non-hierarchical'
                ax.text(0.05, 0.95, hier_text, transform=ax.transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree k', fontsize=12)
        ax.set_ylabel('C(k)', fontsize=12)
        ax.set_title('Clustering Coefficient', fontsize=13)
        ax.grid(True, alpha=0.3)
    
    def plot_temporal_evolution(self, save: bool = True) -> Optional[plt.Figure]:
        if not self.model_results:
            logger.error("No model results data")
            return None
        
        years = sorted(self.model_results.keys())
        
        delta_vals = [self.model_results[y].delta for y in years]
        r2_vals = [self.model_results[y].r2 for y in years]
        mae_vals = [self.model_results[y].mae for y in years]
        rmse_vals = [self.model_results[y].rmse for y in years]
        
        network_sizes = [self.model_results[y].num_nodes for y in years]
        network_edges = [self.model_results[y].num_edges for y in years]
        densities = [self.model_results[y].density for y in years]
        
        assortativity = []
        hierarchy = []
        global_clustering = []
        
        for y in years:
            if y in self.topology_results:
                assort = self.topology_results[y].get('degree_correlations', {}).get('assortativity', np.nan)
                hier = self.topology_results[y].get('clustering', {}).get('hierarchy_correlation', np.nan)
                clust = self.topology_results[y].get('clustering', {}).get('global', np.nan)
            else:
                assort = hier = clust = np.nan
            
            assortativity.append(assort)
            hierarchy.append(hier)
            global_clustering.append(clust)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Temporal Evolution of World Trade Web (2000-2020)', 
                    fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.plot(years, network_sizes, 'o-', linewidth=2, markersize=8, 
                color='darkblue', label='Number of countries')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Number of Countries', fontsize=12)
        ax1.set_title('Network Size', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        ax2 = axes[0, 1]
        ax2.plot(years, densities, 's-', linewidth=2, markersize=8, 
                color='darkgreen', label='Network density')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Network Density', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        ax3 = axes[1, 0]
        ax3.plot(years, delta_vals, '^-', linewidth=2, markersize=8, 
                color='red', label='δ parameter')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('δ', fontsize=12)
        ax3.set_title('Fitness Model Parameter δ', fontsize=13)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        ax4 = axes[1, 1]
        ax4.plot(years, r2_vals, 'o-', linewidth=2, markersize=8, 
                color='purple', label='R²')
        ax4.set_xlabel('Year', fontsize=12)
        ax4.set_ylabel('R²', fontsize=12)
        ax4.set_title('Model Prediction Accuracy (R²)', fontsize=13)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        ax5 = axes[2, 0]
        ax5.plot(years, assortativity, 'd-', linewidth=2, markersize=8, 
                color='orange', label='Assortativity')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Year', fontsize=12)
        ax5.set_ylabel('Assortativity Coefficient', fontsize=12)
        ax5.set_title('Degree Assortativity Evolution', fontsize=13)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=10)
        
        ax6 = axes[2, 1]
        ax6.plot(years, global_clustering, 'v-', linewidth=2, markersize=8, 
                color='brown', label='Global clustering')
        ax6.set_xlabel('Year', fontsize=12)
        ax6.set_ylabel('Clustering Coefficient', fontsize=12)
        ax6.set_title('Global Clustering Coefficient Evolution', fontsize=13)
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save:
            fig_path = os.path.join(self.results_dir, 'figures', 'temporal_evolution.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_all_results(self):
        if self.model_results:
            model_data = {}
            for year, result in self.model_results.items():
                model_data[year] = {
                    'delta': result.delta,
                    'r2': result.r2,
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'num_nodes': result.num_nodes,
                    'num_edges': result.num_edges,
                    'density': result.density,
                    'countries': result.countries,
                    'fitness_values': result.fitness_values,
                    'predicted_degrees': result.predicted_degrees,
                    'actual_degrees': result.actual_degrees
                }
            
            model_file = os.path.join(self.results_dir, 'models', 'fitness_model_results.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            summary_df = pd.DataFrame([r.get_summary_stats() for r in self.model_results.values()])
            csv_file = os.path.join(self.results_dir, 'tables', 'model_summary.csv')
            summary_df.to_csv(csv_file, index=False)
        
        if self.topology_results:
            topology_file = os.path.join(self.results_dir, 'models', 'topology_results.pkl')
            with open(topology_file, 'wb') as f:
                pickle.dump(self.topology_results, f)
        
        self._save_detailed_reports()
        
        print(f"\nAll analysis results saved to: {self.results_dir}")
    
    def _save_detailed_reports(self):
        for year in self.model_results.keys():
            if year in self.topology_results:
                report_dir = os.path.join(self.results_dir, 'reports', str(year))
                os.makedirs(report_dir, exist_ok=True)
                
                model = self.model_results[year]
                df = model.get_comparison_dataframe()
                df.to_csv(os.path.join(report_dir, f'degree_comparison_{year}.csv'), index=False)
                
                topology = self.topology_results[year]
                stats_file = os.path.join(report_dir, f'topology_stats_{year}.txt')
                
                with open(stats_file, 'w', encoding='utf-8') as f:
                    f.write(f"World Trade Web Analysis - Year {year}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write("Network Properties:\n")
                    f.write(f"  Number of countries: {model.num_nodes}\n")
                    f.write(f"  Number of edges: {model.num_edges}\n")
                    f.write(f"  Density: {model.density:.4f}\n")
                    f.write(f"  Model R^2: {model.r2:.4f}\n\n")
                    
                    if 'degree_distribution' in topology:
                        dist = topology['degree_distribution']
                        f.write("Degree Distribution:\n")
                        f.write(f"  Mean degree: {dist['mean']:.2f}\n")
                        f.write(f"  Std degree: {dist['std']:.2f}\n")
                        f.write(f"  Min degree: {dist['min']}\n")
                        f.write(f"  Max degree: {dist['max']}\n")
                        f.write(f"  Skewness: {dist['skewness']:.3f}\n")
                        f.write(f"  Kurtosis: {dist['kurtosis']:.3f}\n\n")
                    
                    if 'degree_correlations' in topology:
                        corr = topology['degree_correlations']
                        f.write("Degree Correlations:\n")
                        f.write(f"  Assortativity: {corr.get('assortativity', 'N/A'):.3f}\n")
                        f.write(f"  Disassortative: {corr.get('disassortative', 'N/A')}\n")
                        if 'power_law_exponent' in corr:
                            f.write(f"  K_nn power law exponent: {corr['power_law_exponent']:.3f}\n")
                        f.write("\n")
                    
                    if 'clustering' in topology:
                        clust = topology['clustering']
                        f.write("Clustering:\n")
                        f.write(f"  Global clustering: {clust['global']:.3f}\n")
                        f.write(f"  Hierarchy correlation: {clust.get('hierarchy_correlation', 'N/A'):.3f}\n")
                        f.write(f"  Hierarchical: {clust.get('hierarchical', 'N/A')}\n")

def main_analysis():
    print("=" * 70)
    print("World Trade Network Fitness Model Analysis")
    print("=" * 70)
    
    try:
        processor = TradeDataProcessor(data_dir='./data')
        networks = processor.load_networks('undirected_wtw_networks.pkl')
        
        if not networks:
            print("No network data found, please run data processing first")
            return
        
        print(f"Year range: {min(networks.keys())} - {max(networks.keys())}")
        
        print("Initializing analyzer...")
        analyzer = FitnessModelAnalyzer(
            networks=networks,
            results_dir='./results'
        )
        
        all_years = sorted(networks.keys())
        analysis_years = [y for y in all_years if y % 5 == 0]
        
        if not analysis_years:
            analysis_years = all_years[-5:]
        
        print(f"\nSelected analysis years: {analysis_years}")
        
        print("\nFitting fitness models...")
        for year in tqdm(analysis_years, desc="Fitting models"):
            result = analyzer.fit_fitness_model(year, verbose=True)
            if result:
                print(f"  {year}: δ={result.delta:.2e}, R²={result.r2:.4f}, "
                      f"{result.num_nodes} countries, {result.num_edges} edges")
        
        print("\nAnalyzing topology...")
        for year in tqdm(analysis_years, desc="Topology analysis"):
            analyzer.analyze_topology(year)
        
        print("\nGenerating analysis figures...")
        for year in analysis_years:
            if year in analyzer.model_results:
                analyzer.plot_comparison_figure(year, save=True)
        
        print("\nGenerating temporal evolution figure...")
        analyzer.plot_temporal_evolution(save=True)
        
        print("\nSaving analysis results...")
        analyzer.save_all_results()
        
        print("\n" + "=" * 70)
        print("Analysis complete!")
        
        print("\nKey results summary:")
        print("-" * 40)
        for year in analysis_years:
            if year in analyzer.model_results:
                model = analyzer.model_results[year]
                print(f"{year}: R²={model.r2:.3f}, δ={model.delta:.2e}, "
                      f"degree range [{min(model.actual_degrees.values())}, "
                      f"{max(model.actual_degrees.values())}]")
        
        return analyzer
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_analysis()