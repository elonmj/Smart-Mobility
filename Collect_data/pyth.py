import requests
import json
import time

# Configuration de la clé API SerpAPI
SERP_API_KEY = "a050f66c64d3a236730d8f30e88a27a1871440a4"

# Liste des routes principales avec leurs intersections critiques
routes_cotonou = {
    "Intersection RNIE1 et Avenue Charles de Gaulle": "Route Nationale 1 Cotonou",
    "Route de l'Aéroport et Avenue du Général De Gaulle": "Route de l'Aéroport Cotonou",
    "Route des Pêches et Adjaha-Athiémé": "Route des Pêches Cotonou"
}

# Fonction pour effectuer une requête via SerpAPI
def fetch_traffic_data(route_name, query):
    url = f"https://serpapi.com/search.json?engine=google_maps&q={query}&api_key={SERP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except ValueError:
            print(f"Erreur de parsing JSON pour {route_name}")
            return None
    else:
        print(f"Erreur HTTP {response.status_code} pour {route_name}")
        print(f"Réponse de l'API : {response.text}")
        return None
    
# Collecte des données pour les 5 minutes à venir
def collect_and_store_data():
    traffic_data = {}

    for route_name, query in routes_cotonou.items():
        print(f"Récupération des données pour : {route_name}")
        data = fetch_traffic_data(route_name, query)
        if data:
            # Extraire des informations pertinentes
            traffic_data[route_name] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "density": data.get("traffic", {}).get("density", 0),
                "average_speed": data.get("traffic", {}).get("average_speed", 0),
                "congestion_level": data.get("traffic", {}).get("congestion", "unknown"),
                "queue_length": data.get("traffic", {}).get("queue_length", 0),
                "contextual_data": {
                    "weather": data.get("weather", {}).get("condition", "unknown"),
                    "road_conditions": data.get("road_conditions", "unknown")
                }
            }
        time.sleep(5)  # Respect des limites de débit de l'API

    # Stockage des données dans un fichier JSON
    with open("traffic_data.json", "w", encoding="utf-8") as json_file:
        json.dump(traffic_data, json_file, indent=4, ensure_ascii=False)

    print("Données collectées et stockées dans 'traffic_data.json'.")

# Exécution de la collecte des données
if __name__ == "__main__":
    collect_and_store_data()