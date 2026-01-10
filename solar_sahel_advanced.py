# Solar Sahel AI - Advanced Edition
# Optimized Site Selection for Solar Infrastructure in Africa
# -----------------------------------------------------------------------------

import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("solar_sahel.log")
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LocationConfig:
    """Configuration for map boundaries and resolution"""
    min_lon: float = -18.0
    max_lon: float = 40.0
    min_lat: float = 10.0
    max_lat: float = 20.0
    resolution: int = 100

@dataclass
class WeightConfig:
    """Importance weights for different factors"""
    sunshine: float = 0.6
    water: float = 0.2
    people: float = 0.1
    roads: float = 0.1

    def validate(self):
        total = self.sunshine + self.water + self.people + self.roads
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, expected 1.0. Normalizing...")
            self.sunshine /= total
            self.water /= total
            self.people /= total
            self.roads /= total

class DataProvider:
    """Manages static data sources"""

    @staticmethod
    def get_country_sunshine() -> pd.DataFrame:
        data = [
            {'country': 'Senegal', 'sun_power': 5.8, 'lat': 14.5, 'lon': -14.5},
            {'country': 'Mali', 'sun_power': 6.1, 'lat': 17.0, 'lon': -4.0},
            {'country': 'Niger', 'sun_power': 6.3, 'lat': 17.0, 'lon': 9.0},
            {'country': 'Chad', 'sun_power': 6.2, 'lat': 15.0, 'lon': 19.0},
            {'country': 'Burkina Faso', 'sun_power': 5.9, 'lat': 12.0, 'lon': -2.0},
            {'country': 'Mauritania', 'sun_power': 6.4, 'lat': 20.0, 'lon': -10.0},
            {'country': 'Sudan', 'sun_power': 6.1, 'lat': 16.0, 'lon': 30.0}
        ]
        return pd.DataFrame(data)

    @staticmethod
    def get_water_sources() -> List[Dict[str, Any]]:
        return [
            {'name': 'Senegal River', 'type': 'river', 'lat': 16.0, 'lon': -12.0, 'size': 3.0},
            {'name': 'Niger River', 'type': 'river', 'lat': 14.0, 'lon': -6.0, 'size': 4.0},
            {'name': 'Lake Chad', 'type': 'lake', 'lat': 13.0, 'lon': 14.0, 'size': 5.0},
            {'name': 'Nile River', 'type': 'river', 'lat': 15.0, 'lon': 32.0, 'size': 3.0},
        ]

    @staticmethod
    def get_population_density() -> Dict[str, int]:
        return {
            'Senegal': 87,
            'Mali': 17,
            'Niger': 20,
            'Chad': 13,
            'Burkina Faso': 76,
            'Mauritania': 5,
            'Sudan': 25
        }

    @staticmethod
    def get_cities() -> List[Dict[str, Any]]:
        return [
            {'name': 'Dakar', 'lat': 14.69, 'lon': -17.45, 'importance': 0.9},
            {'name': 'Bamako', 'lat': 12.65, 'lon': -8.00, 'importance': 0.8},
            {'name': 'Niamey', 'lat': 13.51, 'lon': 2.11, 'importance': 0.7},
            {'name': "N'Djamena", 'lat': 12.13, 'lon': 15.05, 'importance': 0.6},
            {'name': 'Ouagadougou', 'lat': 12.36, 'lon': -1.53, 'importance': 0.7},
        ]

class SolarEngine:
    """Core logic for map generation and scoring"""

    def __init__(self, config: LocationConfig, weights: WeightConfig):
        self.config = config
        self.weights = weights
        self.weights.validate()
        self.data = DataProvider()

        # Grid initialization
        self.lons = np.linspace(config.min_lon, config.max_lon, config.resolution)
        self.lats = np.linspace(config.min_lat, config.max_lat, config.resolution)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lons, self.lats)

    def generate_maps(self) -> Tuple[np.ndarray, ...]:
        """Generates all factor maps"""
        logger.info(f"Generating maps with resolution {self.config.resolution}x{self.config.resolution}")

        sunshine_map = self._compute_sunshine()
        water_map = self._compute_water()
        people_map = self._compute_population(sunshine_map) # Dependency for country matching
        infrastructure_map = self._compute_infrastructure()

        return self.lon_grid, self.lat_grid, sunshine_map, water_map, people_map, infrastructure_map

    def calculate_scores(self, sunshine: np.ndarray, water: np.ndarray,
                        people: np.ndarray, infrastructure: np.ndarray) -> np.ndarray:
        """Calculates the weighted suitability score for each point"""
        logger.info("Calculating composite scores...")

        # Helper for correct global scaling (fixes the bug in original code)
        def scale_global(arr):
            scaler = MinMaxScaler()
            # Reshape to 1D (all pixels), scale, then reshape back
            flat = arr.reshape(-1, 1)
            scaled = scaler.fit_transform(flat)
            return scaled.reshape(arr.shape)

        sun_norm = scale_global(sunshine)
        water_norm = scale_global(water)
        infra_norm = scale_global(infrastructure)

        # Invert people density (lower is better)
        people_norm = 1.0 - scale_global(people)

        total_score = (
            self.weights.sunshine * sun_norm +
            self.weights.water * water_norm +
            self.weights.people * people_norm +
            self.weights.roads * infra_norm
        )

        return total_score

    def classify_zones(self, scores: np.ndarray) -> np.ndarray:
        """Classifies scores into discrete zones"""
        zones = np.zeros_like(scores, dtype=int)
        zones[scores > 0.7] = 3    # Excellent
        zones[(scores > 0.4) & (scores <= 0.7)] = 2  # Good
        zones[(scores > 0.1) & (scores <= 0.4)] = 1  # Okay
        # 0 is Poor/Not Recommended
        return zones

    def _compute_sunshine(self) -> np.ndarray:
        logger.info("Computing solar potential...")
        sunshine_data = self.data.get_country_sunshine()
        sun_map = np.zeros_like(self.lon_grid)

        for _, row in sunshine_data.iterrows():
            dist = self._distance(row['lon'], row['lat'])
            influence = np.exp(-dist / 10.0)
            sun_map += influence * row['sun_power']

        # Realistic scaling
        min_val, max_val = np.min(sun_map), np.max(sun_map)
        if max_val > min_val:
            sun_map = 4.5 + (sun_map - min_val) / (max_val - min_val) * 2.0

        return sun_map

    def _compute_water(self) -> np.ndarray:
        logger.info("Computing water access...")
        water_sources = self.data.get_water_sources()
        water_map = np.zeros_like(self.lon_grid)

        for source in water_sources:
            dist = self._distance(source['lon'], source['lat'])
            access = np.exp(-dist / source['size'])
            water_map += access

        return water_map

    def _compute_population(self, sunshine_map_ref: Any = None) -> np.ndarray:
        logger.info("Computing population density...")
        # Using country centers from sunshine data for simplicity, as per original logic
        sunshine_data = self.data.get_country_sunshine()
        pop_data = self.data.get_population_density()
        pop_map = np.zeros_like(self.lon_grid)

        for _, country in sunshine_data.iterrows():
            name = country['country']
            if name in pop_data:
                density = pop_data[name]
                dist = self._distance(country['lon'], country['lat'])
                clustering = np.exp(-dist / 5.0)
                pop_map += clustering * (density / 100.0)

        return pop_map

    def _compute_infrastructure(self) -> np.ndarray:
        logger.info("Computing infrastructure proximity...")
        cities = self.data.get_cities()
        infra_map = np.zeros_like(self.lon_grid)

        for city in cities:
            dist = self._distance(city['lon'], city['lat'])
            influence = city['importance'] * np.exp(-dist / 3.0)
            infra_map += influence

        return infra_map

    def _distance(self, target_lon: float, target_lat: float) -> np.ndarray:
        """Euclidean distance on the grid"""
        return np.sqrt((self.lon_grid - target_lon)**2 + (self.lat_grid - target_lat)**2)

class Visualizer:
    """Handles all plotting and reporting"""

    @staticmethod
    def save_results(lons, lats, sunshine, water, people, infrastructure, scores, zones):
        logger.info("Generating visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Solar Site Selection - Sahel Region', fontsize=16, fontweight='bold')

        factors = [
            (sunshine, 'Solar Irradiance (kWh/m²/day)', 'YlOrRd', axes[0,0]),
            (water, 'Water Proximity Index', 'Blues', axes[0,1]),
            (people, 'Population Density Index', 'Reds', axes[0,2]),
            (infrastructure, 'Infrastructure Score', 'Purples', axes[1,0]),
            (scores, 'Composite Suitability Score', 'viridis', axes[1,1])
        ]

        for data, title, cmap, ax in factors:
            im = ax.contourf(lons, lats, data, levels=20, cmap=cmap)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax)

        # Zone Map
        zone_colors = ['white', '#FFA500', '#FFFF00', '#008000'] # White, Orange, Yellow, Green
        cmap_zones = ListedColormap(zone_colors)

        ax_zone = axes[1,2]
        ax_zone.contourf(lons, lats, zones, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cmap_zones)
        ax_zone.set_title('Recommended Deployment Zones', fontweight='bold')
        ax_zone.set_xlabel('Longitude')
        ax_zone.set_ylabel('Latitude')

        # Overlay Countries
        country_data = DataProvider.get_country_sunshine()
        ax_zone.plot(country_data['lon'], country_data['lat'], 'ro', markersize=6, alpha=0.8)
        for _, row in country_data.iterrows():
            ax_zone.text(row['lon']+0.5, row['lat'], row['country'], fontsize=8)

        # Legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Unsuitable'),
            Patch(facecolor='#FFA500', edgecolor='gray', label='Marginal'),
            Patch(facecolor='#FFFF00', edgecolor='gray', label='Good'),
            Patch(facecolor='#008000', edgecolor='gray', label='Excellent')
        ]
        ax_zone.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        output_file = 'solar_city_recommendations_advanced.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Map saved to {output_file}")

    @staticmethod
    def print_report(sunshine, scores, zones):
        total_pixels = zones.size
        perfect = np.sum(zones == 3)
        good = np.sum(zones == 2)
        okay = np.sum(zones == 1)

        report = []
        report.append("=" * 50)
        report.append("📋 ADVANCED ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Total Area Units: {total_pixels:,}")
        report.append(f"Excellent Sites:  {perfect:,} ({perfect/total_pixels*100:.1f}%)")
        report.append(f"Good Sites:       {good:,} ({good/total_pixels*100:.1f}%)")
        report.append(f"Marginal Sites:   {okay:,} ({okay/total_pixels*100:.1f}%)")
        report.append("-" * 30)
        report.append(f"Avg Solar Irradiance: {np.mean(sunshine):.2f} kWh/m²/day")
        report.append(f"Max Suitability Score: {np.max(scores):.4f}")

        report_text = "\n".join(report)
        print(report_text)

        with open("analysis_report.txt", "w") as f:
            f.write(report_text)
        logger.info("Report text saved to analysis_report.txt")

def main():
    logger.info("Initializing Solar Sahel AI (Advanced)...")

    config = LocationConfig()
    weights = WeightConfig()
    engine = SolarEngine(config, weights)

    # 1. Generate Maps
    lons, lats, sunshine, water, people, infra = engine.generate_maps()

    # 2. Analyze
    scores = engine.calculate_scores(sunshine, water, people, infra)
    zones = engine.classify_zones(scores)

    # 3. Visualize & Report
    Visualizer.save_results(lons, lats, sunshine, water, people, infra, scores, zones)
    Visualizer.print_report(sunshine, scores, zones)

    logger.info("Process completed successfully.")

if __name__ == "__main__":
    main()
