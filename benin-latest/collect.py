import geopandas as gpd
import pandas as pd
import numpy as np
import os # Pour gérer les chemins de fichiers

# --- Configuration ---
# Chemin vers le DÉCOMPRESSÉ du fichier .shp.zip de Geofabrik pour le Bénin
data_folder = './' # Mettre le chemin vers le dossier contenant les fichiers .shp, .dbf etc.
roads_shapefile_name = 'gis_osm_roads_free_1.shp' # Nom du fichier shape des routes

# Chemin complet vers le fichier shape
shapefile_path = os.path.join(data_folder, roads_shapefile_name)

# Fichier de sortie pour les données traitées (optionnel mais recommandé)
output_csv_path = os.path.join(data_folder, 'benin_roads_processed.csv')
output_geojson_path = os.path.join(data_folder, 'benin_roads_processed.geojson')


# --- Fonction de Classification R(x) ---
# !!! À ADAPTER PAR VOUS EN FONCTION DE VOTRE ANALYSE FINALE DE fclass !!!
# Ceci est un exemple amélioré, traitant les '_link' et séparant 'service'
def classify_road_type_r(fclass_value):
    """
    Classifie un segment de route en fonction de sa fclass OSM.
    Retourne un entier représentant la catégorie R(x).
    Catégories Exemple :
        1: Majeures (Trunk, Primary) + Leurs Links
        2: Secondaires (Secondary, Tertiary) + Leurs Links
        3: Résidentielles/Locales (Residential, Unclassified, Living Street)
        4: Pistes Carrossables (Track, Track_grade*)
        5: Service / Chemins (Service, Path, Footway, Cycleway, Pedestrian, Steps)
        9: Inconnu/Autre (None, unknown, bridleway, etc.)
    """
    if fclass_value is None:
        return 9 # Inconnu si fclass est manquant

    fclass_str = str(fclass_value).lower() # Mettre en minuscule pour la comparaison

    if fclass_str in ['trunk', 'primary', 'motorway', 'trunk_link', 'primary_link', 'motorway_link']:
        return 1
    elif fclass_str in ['secondary', 'tertiary', 'secondary_link', 'tertiary_link']:
        return 2
    elif fclass_str in ['residential', 'unclassified', 'living_street']:
        return 3
    elif 'track' in fclass_str: # Inclut track, track_grade1, track_grade2, etc.
        return 4
    elif fclass_str in ['service', 'path', 'footway', 'cycleway', 'pedestrian', 'steps']:
        return 5
    else: # Inclut 'unknown', 'bridleway' et autres cas non prévus
        return 9

# --- Programme Principal ---
def process_road_data(shp_path, out_csv=None, out_geojson=None):
    """
    Lit le shapefile des routes, calcule la longueur, applique la classification R,
    et sauvegarde les résultats si demandé.
    """
    try:
        print(f"Lecture du fichier Shapefile : {shp_path}")
        try:
            gdf = gpd.read_file(shp_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("Erreur UTF-8, tentative avec l'encodage latin1...")
            gdf = gpd.read_file(shp_path, encoding='latin1')
        print(f"Lecture réussie. Nombre de segments trouvés : {len(gdf)}")
        print("-" * 30)

        # 1. S'assurer que le système de coordonnées est projeté pour calculer la longueur
        print(f"Système de coordonnées initial (CRS) : {gdf.crs}")
        if gdf.crs is None:
            print("ATTENTION : Le CRS n'est pas défini. Impossible de calculer les longueurs précisément.")
            gdf['length_m'] = np.nan # Mettre NaN si pas de CRS
        elif gdf.crs.is_geographic:
            # Si le CRS est géographique (lat/lon), on projette vers un CRS métrique adapté
            # pour le Bénin (UTM Zone 31N: EPSG:32631) pour calculer les longueurs en mètres
            print("Projection vers EPSG:32631 (UTM Zone 31N) pour calcul de longueur...")
            gdf_proj = gdf.to_crs(epsg=32631)
            gdf['length_m'] = gdf_proj.geometry.length
            print("Calcul de longueur terminé.")
        else:
            # Si déjà projeté (moins probable pour les données OSM brutes), on calcule directement
            print("Le CRS est déjà projeté, calcul direct de la longueur...")
            gdf['length_m'] = gdf.geometry.length
            print("Calcul de longueur terminé.")

        print("-" * 30)

        # 2. Appliquer la classification R(x)
        if 'fclass' in gdf.columns:
            print("Application de la classification R(x) basée sur 'fclass'...")
            gdf['R_value'] = gdf['fclass'].apply(classify_road_type_r)
            print("Classification appliquée.")

            # Afficher le décompte des catégories R
            print("\nDécompte des catégories R attribuées :")
            print(gdf['R_value'].value_counts().sort_index())
        else:
            print("ATTENTION : Colonne 'fclass' non trouvée. Impossible d'appliquer la classification R.")
            gdf['R_value'] = 9 # Assigner 'Inconnu' par défaut

        print("-" * 30)

        # 3. Sélectionner les colonnes utiles pour l'analyse
        # Garder osm_id, fclass, R_value, name, ref, oneway, maxspeed, length_m, geometry
        useful_columns = ['osm_id', 'fclass', 'R_value', 'name', 'ref', 'oneway', 'maxspeed', 'length_m']
        # S'assurer que toutes les colonnes existent avant de sélectionner
        final_columns = [col for col in useful_columns if col in gdf.columns] + ['geometry']
        gdf_processed = gdf[final_columns].copy() # .copy() pour éviter SettingWithCopyWarning

        print("Aperçu des données traitées (10 premières lignes) :")
        print(gdf_processed.head(10))
        print("-" * 30)

        # 4. Sauvegarder les résultats (Optionnel)
        if out_csv:
            try:
                print(f"Sauvegarde des données traitées (sans géométrie) en CSV : {out_csv}")
                # Sauvegarder sans la colonne géométrie qui n'est pas gérée par CSV standard
                gdf_processed.drop(columns=['geometry']).to_csv(out_csv, index=False, encoding='utf-8-sig') # utf-8-sig pour Excel
                print("Sauvegarde CSV réussie.")
            except Exception as e_csv:
                print(f"ERREUR lors de la sauvegarde CSV : {e_csv}")
            print("-" * 30)

        if out_geojson:
             try:
                print(f"Sauvegarde des données traitées (avec géométrie) en GeoJSON : {out_geojson}")
                # GeoJSON gère la géométrie
                gdf_processed.to_file(out_geojson, driver='GeoJSON', encoding='utf-8')
                print("Sauvegarde GeoJSON réussie.")
             except Exception as e_geojson:
                print(f"ERREUR lors de la sauvegarde GeoJSON : {e_geojson}")
             print("-" * 30)

        print("Traitement terminé.")
        return gdf_processed # Retourne le GeoDataFrame traité

    except ImportError:
        print("\nERREUR : La bibliothèque GeoPandas n'est pas installée.")
        print("Veuillez l'installer, par exemple avec : pip install geopandas")
        return None
    except FileNotFoundError:
         print(f"\nERREUR : Fichier Shapefile non trouvé : {shp_path}")
         print("Vérifiez le chemin et le nom du fichier.")
         return None
    except Exception as e:
        print(f"\nERREUR inattendue lors du traitement : {e}")
        return None

# --- Exécution ---
if __name__ == "__main__":
    processed_data = process_road_data(shapefile_path,
                                       out_csv=output_csv_path,
                                       out_geojson=output_geojson_path)

    if processed_data is not None:
        # Vous pouvez maintenant utiliser le GeoDataFrame 'processed_data'
        # pour vos analyses ou pour préparer vos simulations.
        # Par exemple, pour trouver toutes les routes primaires (R=1):
        # primary_roads = processed_data[processed_data['R_value'] == 1]
        # print(f"\nTrouvé {len(primary_roads)} segments de routes primaires (R=1).")
        pass