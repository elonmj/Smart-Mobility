import geopandas as gpd
import pandas as pd
import numpy as np # Pour gérer les types numériques et NaN

# --- Configuration ---
# !!! IMPORTANT !!!
# Assurez-vous que ce chemin pointe vers le bon fichier .shp des routes
# (ex: gis_osm_roads_free_1.shp).
shapefile_path = 'gis_osm_roads_free_1.shp' # <--- VÉRIFIEZ / MODIFIEZ SI NECESSAIRE

# --- Fin Configuration ---

try:
    # Lire le fichier Shapefile dans un GeoDataFrame
    print(f"Lecture du fichier Shapefile : {shapefile_path}")
    # L'encodage peut parfois poser problème, essayez 'utf-8' ou 'latin1' si besoin
    try:
        gdf = gpd.read_file(shapefile_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Erreur UTF-8, tentative avec l'encodage latin1...")
        gdf = gpd.read_file(shapefile_path, encoding='latin1')

    print("Lecture réussie.")
    print("-" * 30)

    # 1. Afficher les noms de toutes les colonnes (attributs) disponibles
    print("Attributs disponibles (noms des colonnes) :")
    available_attributes = list(gdf.columns)
    print(available_attributes)
    print("-" * 30)

    # 2. Afficher les 5 premières lignes pour voir des exemples de valeurs
    print("Aperçu des 5 premières lignes (routes et leurs attributs) :")
    # Configurer pandas pour afficher toutes les colonnes si elles sont nombreuses
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000) # Pour élargir l'affichage
    print(gdf.head())
    print("-" * 30)

    # 3. Examiner les valeurs uniques pour l'attribut clé 'fclass'
    if 'fclass' in available_attributes:
        print("Valeurs uniques pour l'attribut 'fclass':")
        unique_fclass = gdf['fclass'].unique()
        print(unique_fclass)

        print("\nDécompte des types 'fclass' (Top 20) :")
        # Afficher le nombre d'occurrences pour mieux comprendre la distribution
        print(gdf['fclass'].value_counts(dropna=False).head(20)) # dropna=False inclut les None
        print("-" * 30)
    else:
        print("L'attribut 'fclass' n'a pas été trouvé dans ce fichier.")
        print("Impossible de déterminer le type de route primaire.")
        print("-" * 30)


    # 4. Analyse détaillée de 'maxspeed'
    if 'maxspeed' in available_attributes:
        print("Analyse de l'attribut 'maxspeed':")

        # Tenter la conversion en numérique, gérant les erreurs
        gdf['maxspeed_numeric'] = pd.to_numeric(gdf['maxspeed'], errors='coerce')

        # Calculs de base
        total_count = len(gdf)
        missing_or_nan_count = gdf['maxspeed_numeric'].isna().sum()
        numeric_count = total_count - missing_or_nan_count

        print(f"Nombre total de segments : {total_count}")
        print(f"Nombre de segments avec 'maxspeed' interprétable numériquement : {numeric_count} ({numeric_count/total_count:.1%})")
        if missing_or_nan_count > 0:
             print(f"Nombre de segments sans 'maxspeed' numérique (ou vide initialement) : {missing_or_nan_count} ({missing_or_nan_count/total_count:.1%})")

        if numeric_count > 0:
            # Filtrer spécifiquement pour les valeurs > 0
            gdf_speed_nonzero = gdf[gdf['maxspeed_numeric'] > 0]['maxspeed_numeric']
            count_nonzero = len(gdf_speed_nonzero)
            print(f"\nNombre de segments avec 'maxspeed' > 0 : {count_nonzero} ({count_nonzero/total_count:.1%})") # Pourcentage par rapport au TOTAL

            if count_nonzero > 0:
                 print("\nStatistiques descriptives pour les 'maxspeed' > 0:")
                 print(gdf_speed_nonzero.describe()) # Calculé uniquement sur les > 0

                 print("\nDécompte des valeurs 'maxspeed' > 0 les plus fréquentes (Top 15) :")
                 print(gdf_speed_nonzero.value_counts().head(15)) # Calculé uniquement sur les > 0
            else:
                print("\nAucune valeur 'maxspeed' strictement positive trouvée.")

            # Compter explicitement les zéros parmi les valeurs numériques
            count_zero = (gdf['maxspeed_numeric'] == 0).sum()
            print(f"\nNombre de segments avec 'maxspeed' = 0 : {count_zero} ({count_zero/total_count:.1%})") # Pourcentage par rapport au TOTAL

        else:
            print("\nAucune valeur 'maxspeed' numérique trouvée après conversion.")

        print("-" * 30)

    else:
        print("L'attribut 'maxspeed' n'a pas été trouvé dans ce fichier.")
        print("-" * 30)

except ImportError:
    print("\nERREUR : La bibliothèque GeoPandas n'est pas installée.")
    print("Veuillez l'installer, par exemple avec : pip install geopandas")
except Exception as e:
    print(f"\nERREUR lors de la lecture ou de l'analyse du fichier : {e}")
    print("Vérifications possibles :")
    print(f"  - Le chemin d'accès '{shapefile_path}' est-il correct ?")
    print("  - Le fichier est-il bien un fichier .shp pour les ROUTES ?")
    print("  - Tous les fichiers associés (.shp, .dbf, .shx, .prj...) sont-ils présents dans le même dossier ?")
    print("  - Le fichier n'est-il pas corrompu ?")
    print("  - Essayez de spécifier un autre encodage si l'erreur est liée aux caractères (ex: encoding='latin1').")