import pandas as pd
import numpy as np

# --- CONFIGURATION ---
NOM_FICHIER_DONNEES = 'donnees_test_24h.csv'

print("-" * 60)
print("--- DÉBUT DE L'ANALYSE APPROFONDIE DE LA VOLATILITÉ DU TRAFIC ---")
print("-" * 60)

# --- ÉTAPE 0: CHARGEMENT ET PRÉPARATION ---
try:
    df = pd.read_csv(NOM_FICHIER_DONNEES)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    print(f"✅ Fichier '{NOM_FICHIER_DONNEES}' chargé avec succès.\n")
except FileNotFoundError:
    print(f"❌ ERREUR: Fichier non trouvé.")
    exit()

# --- 1. ISOLATION DES PÉRIODES DE POINTE ---
df_pointe_matin = df[(df['hour'] >= 6) & (df['hour'] < 10)]
df_pointe_soir = df[(df['hour'] >= 17) & (df['hour'] < 22)]

print("--- 1. ANALYSE DE LA VOLATILITÉ PENDANT LES HEURES DE POINTE ---")
print("L'écart-type (std) mesure à quel point la vitesse a fluctué.")
print("Un 'std' élevé signifie beaucoup de changements, un 'std' faible signifie une vitesse stable.\n")

print("Volatilité par rue (Pointe du Matin, 06h-10h) :")
print(df_pointe_matin.groupby('name')['current_speed'].agg(['mean', 'std', 'min', 'max']).round(2))
print("\nVolatilité par rue (Pointe du Soir, 17h-22h) :")
print(df_pointe_soir.groupby('name')['current_speed'].agg(['mean', 'std', 'min', 'max']).round(2))

# --- 2. ANALYSE DES "SAUTS" DE VITESSE ENTRE LES CYCLES ---
print("\n--- 2. ANALYSE DES 'SAUTS' DE VITESSE ENTRE LES CYCLES DE COLLECTE ---")
print("On calcule la différence de vitesse entre une mesure et la suivante pour chaque segment.")
print("Un 'saut' important signifierait qu'on rate des changements rapides.\n")

# Calcul de la différence de vitesse pour chaque segment
df = df.sort_values(by=['u', 'v', 'timestamp'])
df['speed_diff'] = df.groupby(['u', 'v'])['current_speed'].diff().abs()

# Statistiques sur ces différences, en ignorant les NaNs
speed_diff_stats = df['speed_diff'].describe().round(2)
print("Statistiques sur les sauts de vitesse (en km/h) entre deux cycles consécutifs :")
print(speed_diff_stats)

print(f"\nANALYSE : 95% des changements de vitesse entre deux mesures sont inférieurs à {df['speed_diff'].quantile(0.95):.2f} km/h.")
print(f"Le saut de vitesse maximum observé entre deux mesures consécutives (3-5 min) est de {df['speed_diff'].max():.2f} km/h.")

# --- 3. VISUALISATION DE LA COURBE DE TRAFIC JOURNALIÈRE ---
print("\n--- 3. VISUALISATION DE LA COURBE DE TRAFIC JOURNALIÈRE ---")
print("Vitesse moyenne globale, heure par heure, pour voir les pics de congestion :")

# Calcul de la vitesse moyenne par heure, en s'assurant que l'index est complet de 0 à 23h
speed_by_hour = df.groupby('hour')['current_speed'].mean().reindex(np.arange(24)).round(1)
print(speed_by_hour.to_string())


print("\n" + "-" * 60)
print("--- FIN DE L'ANALYSE APPROFONDIE ---")
print("-" * 60)