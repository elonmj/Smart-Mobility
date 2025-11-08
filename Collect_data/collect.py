# -*- coding: utf-8 -*-
import osmnx as ox
import pandas as pd

# --- Étape 1: Téléchargement des données (inchangé) ---
point_of_interest = (6.431108, 3.423805)
print("--- Téléchargement du réseau routier pour Victoria Island ---")
G = ox.graph_from_point(point_of_interest, dist=1500, network_type='drive')
edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
print("Téléchargement terminé.")

# --- Étape 2: Nettoyage et préparation des données (inchangé) ---
edges_gdf = edges_gdf.reset_index()
if 'name' in edges_gdf.columns:
    edges_gdf['name_clean'] = edges_gdf['name'].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
named_edges_gdf = edges_gdf.dropna(subset=['name_clean'])

# --- Étape 3: Sélection de TOUTES les rues du corridor (inchangé) ---
print("\n--- Sélection des segments des rues du corridor pour la qualification ---")
streets_to_qualify = [
    "Akin Adesola Street",
    "Adeola Odeku Street",
    "Saka Tinubu Street",
    "Ahmadu Bello Way"
]
corridor_gdf = named_edges_gdf[named_edges_gdf['name_clean'].isin(streets_to_qualify)].copy()
print(f"Total de {len(corridor_gdf)} segments routiers sélectionnés.")

# --- Étape 4: Création du fichier Excel avec les valeurs pré-remplies ---
print("--- Création du fichier Excel 'fichier_de_travail_complet.xlsx' ---")

# === MODIFICATION ICI: On assigne directement les valeurs au lieu de laisser vide ===
corridor_gdf['lanes_manual'] = 3
corridor_gdf['Rx_manual'] = 2
corridor_gdf['maxspeed_manual_kmh'] = '' # On laisse vide pour le moment

# Organisation des colonnes (inchangé)
all_available_columns = corridor_gdf.columns.tolist()
manual_columns = ['lanes_manual', 'Rx_manual', 'maxspeed_manual_kmh']
auto_columns = [col for col in all_available_columns if col not in manual_columns]
final_output_columns = auto_columns + manual_columns

file_path = 'fichier_de_travail_complet.xlsx'
corridor_gdf[final_output_columns].to_excel(file_path, index=False)

print(f"\n✅ SUCCÈS ! Le fichier '{file_path}' a été créé.")
print("   Les colonnes 'lanes_manual' et 'Rx_manual' ont été pré-remplies avec les valeurs 3 et 2.")