import requests
import time
import pandas as pd
from shapely import wkt
from datetime import datetime
import os

# --- CONFIGURATION DU TEST ---
VOTRE_CLE_API_TOMTOM = "3CXyWnnL1XPIRbF8N6X1qOTJrC7Mv1Ns" # Mettez votre clé ici
FICHIER_CORRIDOR = 'fichier_de_travail_complet.xlsx'
FICHIER_SORTIE_CSV = 'donnees_test_24h.csv'

# === Liste des 10 segments stratégiques pour le test de 24h ===
SEGMENTS_CLES_A_TESTER = [
    (31674707, 31700906), (31700878, 5109668001), (95636908, 4708819230), (36240967, 31674708),
    (35723955, 1235946207), (35723960, 35723963), (2339926118, 2339926116), (36240962, 95636908),
    (168581819, 1598717904), (4707949582, 95637019)
]

# --- CHARGEMENT ET PRÉPARATION DU TEST ---
print("--- Initialisation du COLLECTEUR DE TEST (1 clé API) ---")
try:
    df_corridor_full = pd.read_excel(FICHIER_CORRIDOR)
    df_corridor_full['geometry'] = df_corridor_full['geometry'].apply(wkt.loads)
    df_corridor = df_corridor_full[df_corridor_full.apply(lambda row: (row['u'], row['v']) in SEGMENTS_CLES_A_TESTER, axis=1)]
    print(f"{len(df_corridor)} segments sélectionnés. Stratégie 'Saturation Intelligente' activée.")
except Exception as e:
    print(f"Erreur critique lors du chargement : {e}")
    exit()

if "VOTRE_UNIQUE_CLE_API" in VOTRE_CLE_API_TOMTOM:
    print("\n!!! ATTENTION !!! Veuillez insérer votre clé API TomTom.")
    exit()

print(f"La collecte de test va commencer. Sortie dans '{FICHIER_SORTIE_CSV}'.")
print("-" * 50)

# --- BOUCLE DE COLLECTE ---
while True:
    try:
        heure_actuelle = datetime.now().hour
        intervalle_secondes = 180 if (6 <= heure_actuelle < 10) or (16 <= heure_actuelle < 22) else (300 if (10 <= heure_actuelle < 16) else 900)
        
        timestamp_cycle = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n--- Cycle de test à {timestamp_cycle} (Intervalle: {intervalle_secondes/60:.0f} min) ---")
        
        results_list = []
        for index, segment in df_corridor.iterrows():
            try:
                geom_line = segment['geometry']
                point_milieu = geom_line.interpolate(0.5)
                lat, lon = point_milieu.y, point_milieu.x
                url_flow = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/15/json?point={lat},{lon}&key={VOTRE_CLE_API_TOMTOM}"
                response = requests.get(url_flow, timeout=15)
                response.raise_for_status()
                data = response.json()
                flow_info = data.get('flowSegmentData')
                result_dict = {'timestamp': timestamp_cycle, 'u': segment['u'], 'v': segment['v'], 'name': segment['name_clean'], 'current_speed': None, 'freeflow_speed': None, 'confidence': None}
                if flow_info:
                    result_dict['current_speed'] = flow_info.get('currentSpeed')
                    result_dict['freeflow_speed'] = flow_info.get('freeFlowSpeed')
                    result_dict['confidence'] = flow_info.get('confidence')
                results_list.append(result_dict)
                time.sleep(0.1)
            except requests.exceptions.HTTPError as err:
                # Si l'erreur est un 403, c'est probablement que le quota est dépassé
                if err.response.status_code == 403:
                    print("  ! ERREUR 403: Quota API probablement dépassé. Mise en pause du script pour 1 heure.")
                    time.sleep(3600) # On attend une heure avant de réessayer
                continue
            except requests.exceptions.RequestException:
                continue

        if results_list:
            df_results = pd.DataFrame(results_list)
            file_exists = os.path.exists(FICHIER_SORTIE_CSV)
            df_results.to_csv(FICHIER_SORTIE_CSV, mode='a', header=not file_exists, index=False)
            print(f"--- Cycle de test terminé. {len(df_results)} résultats sauvegardés. ---")
            
        print(f"Prochain cycle dans {intervalle_secondes / 60:.0f} minutes...")
        time.sleep(intervalle_secondes)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Erreur inattendue : {e}. Redémarrage dans 1 minute.")
        time.sleep(60)