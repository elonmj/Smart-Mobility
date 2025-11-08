import os
import time
import json
from datetime import datetime
import googlemaps
import pandas as pd
from config import GOOGLE_MAPS_API_KEY, COTONOU_COORDINATES, DATA_DIR

class TrafficDataCollector:
    def __init__(self):
        self.gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
    
    def get_traffic_data(self):
        """Collect traffic data for predefined routes in Cotonou"""
        # Define key locations in Cotonou
        key_locations = [
            ("Dantokpa Market", {"lat": 6.3739, "lng": 2.4301}),
            ("Airport", {"lat": 6.3572, "lng": 2.3847}),
            ("Cadjehoun", {"lat": 6.3721, "lng": 2.3883}),
            ("Akpakpa", {"lat": 6.3639, "lng": 2.4439})
        ]
        
        traffic_data = []
        timestamp = datetime.now()
        
        # Collect data between key locations
        for origin_name, origin_coords in key_locations:
            for dest_name, dest_coords in key_locations:
                if origin_name != dest_name:
                    try:
                        # Get directions with traffic information
                        directions = self.gmaps.directions(
                            origin=origin_coords,
                            destination=dest_coords,
                            departure_time=timestamp,
                            traffic_model='best_guess'
                        )
                        
                        if directions:
                            route_data = {
                                'timestamp': timestamp.isoformat(),
                                'origin': origin_name,
                                'destination': dest_name,
                                'duration_in_traffic': directions[0]['legs'][0].get('duration_in_traffic', {}).get('value'),
                                'normal_duration': directions[0]['legs'][0]['duration']['value'],
                                'distance': directions[0]['legs'][0]['distance']['value']
                            }
                            traffic_data.append(route_data)
                            
                    except Exception as e:
                        print(f"Error collecting data for {origin_name} to {dest_name}: {str(e)}")
        
        return traffic_data
    
    def save_data(self, data):
        """Save traffic data to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        filename = os.path.join(DATA_DIR, f'traffic_data_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Data saved to {filename}")