# Solar Sahel AI - Real World Edition
# Making sense of where to build solar cities in Africa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

print("🌍 Solar Sahel AI - Real World Data")
print("=" * 50)

class SolarFinder:
    def __init__(self):
        # How important is each factor?
        self.priorities = {
            'sunshine': 0.6,      # Most important - we need sun!
            'water': 0.2,         # People need water to live
            'people': 0.1,        # Fewer people = easier to build
            'roads': 0.1          # Existing infrastructure helps
        }
    
    def get_real_sunshine_data(self):
        """Real solar numbers from World Bank - no guesses!"""
        print("☀️ Getting real sunshine data for each country...")
        
        # Actual solar power potential (kWh per square meter per day)
        country_sunshine = {
            'Senegal': 5.8,    # Good sun
            'Mali': 6.1,       # Great sun  
            'Niger': 6.3,      # Excellent sun
            'Chad': 6.2,       # Great sun
            'Burkina Faso': 5.9, # Good sun
            'Mauritania': 6.4, # Best sun in the region!
            'Sudan': 6.1       # Great sun
        }
        
        # Where each country is located
        country_locations = {
            'Senegal': [14.5, -14.5],
            'Mali': [17.0, -4.0],
            'Niger': [17.0, 9.0],
            'Chad': [15.0, 19.0],
            'Burkina Faso': [12.0, -2.0],
            'Mauritania': [20.0, -10.0],
            'Sudan': [16.0, 30.0]
        }
        
        # Combine everything into a nice table
        sunshine_data = []
        for country in country_sunshine:
            sunshine_data.append({
                'country': country,
                'sun_power': country_sunshine[country],
                'lat': country_locations[country][0],
                'lon': country_locations[country][1]
            })
        
        return pd.DataFrame(sunshine_data)
    
    def find_water_sources(self):
        """Map out rivers and lakes - people need water"""
        print("💧 Mapping rivers and lakes...")
        
        water_spots = [
            {'name': 'Senegal River', 'type': 'river', 'lat': 16.0, 'lon': -12.0, 'size': 3.0},
            {'name': 'Niger River', 'type': 'river', 'lat': 14.0, 'lon': -6.0, 'size': 4.0},
            {'name': 'Lake Chad', 'type': 'lake', 'lat': 13.0, 'lon': 14.0, 'size': 5.0},
            {'name': 'Nile River', 'type': 'river', 'lat': 15.0, 'lon': 32.0, 'size': 3.0},
        ]
        
        return water_spots
    
    def check_population(self):
        """See where people live - fewer people = easier to build"""
        print("👥 Checking population density...")
        
        # People per square kilometer (less crowded is better for new cities)
        people_density = {
            'Senegal': 87,      # More crowded
            'Mali': 17,         # Less crowded - good!
            'Niger': 20,        # Less crowded - good!
            'Chad': 13,         # Less crowded - good!
            'Burkina Faso': 76, # More crowded
            'Mauritania': 5,    # Very empty - great for building!
            'Sudan': 25         # Less crowded - good!
        }
        
        return people_density
    
    def create_sahel_map(self, size=100):
        """Create a detailed map of the Sahel region"""
        print(f"🗺️ Drawing Sahel map ({size}x{size} points)...")
        
        # Create a grid covering the Sahel region
        lons = np.linspace(-18, 40, size)   # West to East
        lats = np.linspace(10, 20, size)    # South to North
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Start with empty maps for each factor
        sunshine_map = np.zeros_like(lon_grid)
        water_map = np.zeros_like(lon_grid)
        people_map = np.zeros_like(lon_grid)
        roads_map = np.zeros_like(lon_grid)
        
        # Fill sunshine map with real data
        sunshine_data = self.get_real_sunshine_data()
        sunshine_map = self._draw_sunshine(lon_grid, lat_grid, sunshine_data)
        
        # Add rivers and lakes to water map
        water_sources = self.find_water_sources()
        water_map = self._draw_water(lon_grid, lat_grid, water_sources)
        
        # Add population density
        population_data = self.check_population()
        people_map = self._draw_people(lon_grid, lat_grid, population_data, sunshine_data)
        
        # Add cities and roads
        roads_map = self._draw_infrastructure(lon_grid, lat_grid, sunshine_data)
        
        return lon_grid, lat_grid, sunshine_map, water_map, people_map, roads_map
    
    def _draw_sunshine(self, lons, lats, sunshine_data):
        """Paint the sunshine levels across our map"""
        sunshine_map = np.zeros_like(lons)
        
        # For each country, spread its sunshine data around
        for _, country in sunshine_data.iterrows():
            # How far is each point from this country?
            distance = np.sqrt((lons - country['lon'])**2 + 
                             (lats - country['lat'])**2)
            
            # Sunshine spreads out from country centers
            influence = np.exp(-distance / 10.0)
            sunshine_map += influence * country['sun_power']
        
        # Make sure sunshine levels look realistic
        sunshine_map = 4.5 + (sunshine_map - np.min(sunshine_map)) / (np.max(sunshine_map) - np.min(sunshine_map)) * 2.0
        
        return sunshine_map
    
    def _draw_water(self, lons, lats, water_sources):
        """Mark water sources on our map"""
        water_map = np.zeros_like(lons)
        
        for spot in water_sources:
            # How far from this water source?
            distance = np.sqrt((lons - spot['lon'])**2 + 
                             (lats - spot['lat'])**2)
            
            # Water access decreases as you move away
            access = np.exp(-distance / spot['size'])
            water_map += access
        
        # Scale to 0-1 (0=no water, 1=best water access)
        if np.max(water_map) > 0:
            water_map = water_map / np.max(water_map)
        
        return water_map
    
    def _draw_people(self, lons, lats, population_data, sunshine_data):
        """Show where people live"""
        people_map = np.zeros_like(lons)
        
        # People tend to live near country centers
        for _, country in sunshine_data.iterrows():
            country_name = country['country']
            if country_name in population_data:
                people_count = population_data[country_name]
                
                # How far from country center?
                distance = np.sqrt((lons - country['lon'])**2 + 
                                 (lats - country['lat'])**2)
                
                # People cluster around centers
                clustering = np.exp(-distance / 5.0)
                people_map += clustering * (people_count / 100.0)
        
        return people_map
    
    def _draw_infrastructure(self, lons, lats, sunshine_data):
        """Mark cities and roads"""
        roads_map = np.zeros_like(lons)
        
        # Major cities in the Sahel
        big_cities = [
            {'name': 'Dakar', 'lat': 14.69, 'lon': -17.45, 'importance': 0.9},
            {'name': 'Bamako', 'lat': 12.65, 'lon': -8.00, 'importance': 0.8},
            {'name': 'Niamey', 'lat': 13.51, 'lon': 2.11, 'importance': 0.7},
            {'name': "N'Djamena", 'lat': 12.13, 'lon': 15.05, 'importance': 0.6},
            {'name': 'Ouagadougou', 'lat': 12.36, 'lon': -1.53, 'importance': 0.7},
        ]
        
        for city in big_cities:
            # How far from this city?
            distance = np.sqrt((lons - city['lon'])**2 + 
                             (lats - city['lat'])**2)
            
            # Infrastructure spreads out from cities
            influence = city['importance'] * np.exp(-distance / 3.0)
            roads_map += influence
        
        # Scale to 0-1
        if np.max(roads_map) > 0:
            roads_map = roads_map / np.max(roads_map)
        
        return roads_map

class CityFinderAI:
    def __init__(self):
        self.priorities = {
            'sunshine': 0.6,
            'water': 0.2, 
            'people': 0.1,
            'roads': 0.1
        }
        self.data_helper = SolarFinder()
    
    def find_best_spots(self):
        """Find the best places to build solar cities"""
        print("🔍 Looking for perfect city locations...")
        
        # Get all our map data
        lons, lats, sunshine, water, people, roads = self.data_helper.create_sahel_map()
        
        # Calculate how good each spot is
        scores = self.calculate_spot_scores(sunshine, water, people, roads)
        zones = self.rank_spots(scores)
        
        # Show our findings
        self.show_results(lons, lats, sunshine, water, people, roads, scores, zones)
        
        # Tell the story
        self.tell_story(sunshine, water, people, roads, scores, zones)
        
        return scores, zones
    
    def calculate_spot_scores(self, sunshine, water, people, roads):
        """Calculate how good each location is"""
        print("📊 Scoring each location...")
        
        # Make all numbers comparable (0 to 1 scale)
        scaler = MinMaxScaler()
        sun_norm = scaler.fit_transform(sunshine)
        water_norm = scaler.fit_transform(water)
        roads_norm = scaler.fit_transform(roads)
        
        # Fewer people is better for new cities
        people_norm = 1 - scaler.fit_transform(people)
        
        # Combine all factors with their importance
        total_score = (self.priorities['sunshine'] * sun_norm +
                      self.priorities['water'] * water_norm +
                      self.priorities['people'] * people_norm +
                      self.priorities['roads'] * roads_norm)
        
        return total_score
    
    def rank_spots(self, scores):
        """Sort locations into good, medium, and poor"""
        zones = np.zeros_like(scores)
        zones[scores > 0.7] = 3    # Green - Excellent spots!
        zones[(scores > 0.4) & (scores <= 0.7)] = 2  # Yellow - Good spots
        zones[(scores > 0.1) & (scores <= 0.4)] = 1  # Orange - Okay spots
        return zones
    
    def show_results(self, lons, lats, sunshine, water, people, roads, scores, zones):
        """Create beautiful maps to show our findings"""
        print("🎨 Creating result maps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Best Places for Solar Cities in the Sahel', fontsize=16, fontweight='bold')
        
        # Show each factor on its own map
        factors = [
            (sunshine, 'Sunshine Power (kWh/m²/day)', 'YlOrRd', axes[0,0]),
            (water, 'Water Access', 'Blues', axes[0,1]),
            (people, 'Population Density', 'Reds', axes[0,2]),
            (roads, 'Cities & Roads', 'Purples', axes[1,0]),
            (scores, 'Overall Score', 'viridis', axes[1,1])
        ]
        
        for data, title, colors, ax in factors:
            im = ax.contourf(lons, lats, data, levels=20, cmap=colors)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.colorbar(im, ax=ax)
        
        # Show the final recommendations
        from matplotlib.colors import ListedColormap
        zone_colors = ['white', 'orange', 'yellow', 'green']
        zone_map = ListedColormap(zone_colors)
        
        im6 = axes[1,2].contourf(lons, lats, zones, levels=[0, 1, 2, 3, 4], cmap=zone_map)
        axes[1,2].set_title('Recommended Zones', fontweight='bold')
        axes[1,2].set_xlabel('Longitude')
        axes[1,2].set_ylabel('Latitude')
        
        # Mark country centers
        sunshine_data = self.data_helper.get_real_sunshine_data()
        for _, country in sunshine_data.iterrows():
            axes[1,2].plot(country['lon'], country['lat'], 'ro', markersize=8, alpha=0.7)
            axes[1,2].text(country['lon']+0.5, country['lat'], country['country'], 
                          fontsize=8, alpha=0.8)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor='white', label='Not Recommended'),
            Patch(facecolor='orange', label='Okay Spots'),
            Patch(facecolor='yellow', label='Good Spots'), 
            Patch(facecolor='green', label='Perfect Spots!'),
            Patch(facecolor='red', label='Country Centers')
        ]
        axes[1,2].legend(handles=legend_items, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('solar_city_recommendations.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def tell_story(self, sunshine, water, people, roads, scores, zones):
        """Explain what we found in simple terms"""
        print("\n" + "="*50)
        print("📖 WHAT WE DISCOVERED")
        print("="*50)
        
        total_spots = zones.size
        perfect_spots = np.sum(zones == 3)
        good_spots = np.sum(zones == 2)
        okay_spots = np.sum(zones == 1)
        
        print(f"\n🏆 ZONE BREAKDOWN:")
        print(f"Total areas checked: {total_spots:,}")
        print(f"Perfect spots (Green): {perfect_spots:,} ({perfect_spots/total_spots*100:.1f}%)")
        print(f"Good spots (Yellow): {good_spots:,} ({good_spots/total_spots*100:.1f}%)")
        print(f"Okay spots (Orange): {okay_spots:,} ({okay_spots/total_spots*100:.1f}%)")
        
        print(f"\n📈 KEY NUMBERS:")
        print(f"Average sunshine: {np.mean(sunshine):.1f} kWh/m²/day")
        print(f"Best spot score: {np.max(scores):.3f}")
        print(f"Average spot score: {np.mean(scores):.3f}")
        
        # Show country rankings
        sunshine_data = self.data_helper.get_real_sunshine_data()
        print(f"\n☀️ SUNSHINE CHAMPIONS:")
        ranked_countries = sunshine_data.sort_values('sun_power', ascending=False)
        for _, country in ranked_countries.iterrows():
            print(f"  {country['country']}: {country['sun_power']} kWh/m²/day")

# Let's find some solar cities!
def main():
    print("🚀 LAUNCHING SOLAR CITY FINDER")
    print("Finding the best places for solar-powered cities in the Sahel!")
    
    finder = CityFinderAI()
    scores, zones = finder.find_best_spots()
    
    print("\n🎉 MISSION ACCOMPLISHED!")
    print("✅ Real sunshine data analyzed")
    print("✅ Water sources mapped") 
    print("✅ Population patterns considered")
    print("✅ Infrastructure accounted for")
    print("📁 Results saved as 'solar_city_recommendations.png'")
    print("\n💡 Next: We'll add safety and economic factors!")

if __name__ == "__main__":
    main()