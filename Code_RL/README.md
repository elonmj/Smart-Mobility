# Système de Contrôle de Signalisation par Apprentissage par Renforcement
# Adaptation Victoria Island Lagos, Nigeria

## 📋 Vue d'ensemble

Ce projet implémente un système complet de contrôle de feux de signalisation basé sur l'apprentissage par renforcement (RL), spécialement adapté pour le corridor Victoria Island à Lagos, Nigeria. Le système utilise un algorithme Deep Q-Network (DQN) pour optimiser la gestion du trafic dans un environnement urbain dense avec un mix de véhicules (voitures et motos).

## 🌍 Contexte - Victoria Island Lagos

Victoria Island est un quartier central d'affaires de Lagos caractérisé par :
- **Trafic dense multi-modal** : 35% de motos, 65% de voitures
- **Intersections complexes** : Akin Adesola Street x Adeola Odeku Street
- **Routes hiérarchisées** : Primary (3 voies, 50 km/h), Secondary (2 voies, 40 km/h)
- **Comportements de conduite ouest-africains** : Gap-filling, dépassements fréquents

## 🏗️ Architecture du Système

### Composants Principaux

```
┌─────────────────────────────────────────────────────────┐
│                    Agent DQN                           │
│                (Stable-Baselines3)                     │
└─────────────┬───────────────────────────────────────────┘
              │ Actions (0: maintenir, 1: changer)
              ▼
┌─────────────────────────────────────────────────────────┐
│            Environnement RL Gymnasium                  │
│     (TrafficSignalEnv - Normalisation & Récompenses)   │
└─────────────┬───────────────────────────────────────────┘
              │ États/Observations
              ▼
┌─────────────────────────────────────────────────────────┐
│           Contrôleur de Signalisation                  │
│        (Gestion phases, timing, sécurité)              │
└─────────────┬───────────────────────────────────────────┘
              │ Commandes signaux
              ▼
┌─────────────────────────────────────────────────────────┐
│              Simulateur ARZ                           │
│    (Modèle trafic multi-classe + Client Mock)          │
└─────────────────────────────────────────────────────────┘
```

### 1. Simulateur ARZ (Arz-Zuriguel Model)
- **Modèle de trafic multi-classe** supportant motos et voitures
- **Paramètres réalistes** : vitesses libres différenciées, densités maximales
- **Client Mock** pour tests sans simulateur externe
- **Interface REST** pour intégration SUMO/CARLA future

### 2. Environnement Gymnasium
- **Espace d'observation** : 43 dimensions (densités, vitesses, files, timing phases)
- **Espace d'actions** : 2 actions discrètes (maintenir/changer phase)
- **Fonction de récompense multi-objectif** :
  - Minimisation temps d'attente (poids 1.2)
  - Réduction longueur files d'attente (poids 0.6) 
  - Maximisation débit (poids 1.0)
  - Pénalité changements fréquents (poids 0.1)

### 3. Contrôleur de Signalisation
- **Gestion des phases** : Nord-Sud / Est-Ouest
- **Contraintes de sécurité** : Temps verts min/max, transitions sûres
- **Timing adapté Lagos** : Vert min 15s, max 90s, jaune 4s

### 4. Agent DQN
- **Réseau de neurones** : Architecture adaptée aux 43 observations
- **Exploration** : ε-greedy avec décroissance
- **Mémoire de replay** : Buffer d'expériences pour stabilité
- **Target network** : Mise à jour périodique pour stabilité

## 📂 Structure du Projet

```
Code_RL/
├── 📁 src/                    # Code source principal
│   ├── 📁 arz/               # Simulateur ARZ multi-classe
│   │   ├── arz_model.py      # Modèle trafic Arz-Zuriguel étendu
│   │   └── traffic_generator.py # Génération scénarios trafic
│   ├── 📁 endpoint/          # Clients simulateur
│   │   ├── client.py         # Client REST + Mock
│   │   └── mock_client.py    # Simulateur mock intégré
│   ├── 📁 env/              # Environnement RL Gymnasium
│   │   └── traffic_signal_env.py # Env principal avec normalisation
│   ├── 📁 rl/               # Algorithmes apprentissage
│   │   ├── train_dqn.py     # Entraînement DQN avec évaluation
│   │   └── baseline.py      # Baselines (fixe, adaptatif)
│   ├── 📁 signals/          # Contrôle signalisation
│   │   └── controller.py    # Contrôleur phases + sécurité
│   └── 📁 utils/            # Utilitaires
│       ├── config.py        # Chargement configurations YAML
│       └── evaluation.py    # Métriques et évaluation
│
├── 📁 configs/              # Configurations système
│   ├── endpoint.yaml        # Config client simulateur
│   ├── network.yaml         # Réseau générique 4-branches
│   ├── network_real.yaml    # Réseau réel Victoria Island
│   ├── env.yaml            # Environnement RL générique
│   ├── env_lagos.yaml      # Environnement adapté Lagos
│   ├── signals.yaml        # Signalisation générique
│   ├── signals_lagos.yaml  # Signalisation adaptée Lagos
│   ├── traffic_lagos.yaml  # Paramètres trafic Lagos
│   └── lagos_master.yaml   # Configuration maître Lagos
│
├── 📁 data/                # Données réelles
│   ├── donnees_vitesse_historique.csv    # Données vitesses
│   └── fichier_de_travail_corridor.csv   # Corridor Victoria Island
│
├── 📁 scripts/             # Scripts utilitaires
│   ├── demo.py            # Démonstrations interactives
│   ├── train.py           # Script entraînement principal
│   ├── analyze_corridor.py # Analyse données corridor
│   ├── adapt_lagos.py     # Génération configs Lagos
│   └── test_lagos.py      # Tests configuration Lagos
│
├── 📁 tests/              # Tests unitaires
│   └── test_components.py # Tests composants système
│
└── 📁 docs/              # Documentation
    ├── plan_code.md      # Architecture détaillée
    └── implementation/   # Documentation technique
```

## 🔧 Installation et Configuration

### Prérequis Système
- **Python 3.8+** 
- **Windows 10/11** (testé) ou Linux
- **RAM** : 4GB minimum, 8GB recommandé
- **Stockage** : 2GB d'espace libre

### Installation des Dépendances

```bash
# Cloner le projet
git clone [URL_REPO]
cd Code_RL

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales
```
# Apprentissage par renforcement
stable-baselines3==2.0.0
gymnasium==0.29.0
torch>=1.13.0

# Calcul scientifique
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Configuration et utilitaires
pyyaml>=6.0
tqdm>=4.62.0
matplotlib>=3.5.0

# Tests
pytest>=7.0.0
```

## 🚀 Utilisation

### 1. Tests et Validation

```bash
# Test configuration Lagos
python test_lagos.py

# Démonstrations interactives
python demo.py 1    # Composants de base
python demo.py 2    # Environnement RL
python demo.py 3    # Entraînement rapide

# Tests unitaires
pytest tests/
```

### 2. Entraînement du Modèle

```bash
# Entraînement avec configuration Lagos
python train.py --config lagos --use-mock --timesteps 10000

# Entraînement configuration générique
python train.py --use-mock --timesteps 5000

# Avec évaluation étendue
python train.py --config lagos --use-mock --timesteps 20000 --eval-episodes 20
```

### 3. Analyse des Données

```bash
# Analyse corridor Victoria Island
python analyze_corridor.py

# Génération configuration Lagos
python adapt_lagos.py
```

## ⚙️ Configuration Lagos

Le système utilise une configuration spécialement adaptée au contexte de Victoria Island :

### Configuration Trafic (`traffic_lagos.yaml`)
```yaml
traffic:
  context: "Victoria Island Lagos"
  vehicle_composition:
    motorcycles: 0.35    # 35% motos
    cars: 0.65          # 65% voitures
  
  # Paramètres motos
  motorcycles:
    v_free: 32.0        # km/h vitesse libre
    rho_max: 250        # véh/km densité max
    
  # Paramètres voitures  
  cars:
    v_free: 28.0        # km/h vitesse libre
    rho_max: 120        # véh/km densité max
```

### Configuration Environnement (`env_lagos.yaml`)
```yaml
environment:
  dt_decision: 10.0     # Décisions toutes les 10s
  
  reward:
    w_wait_time: 1.2    # Poids temps attente (élevé)
    w_queue_length: 0.6 # Poids longueur files
    w_throughput: 1.0   # Poids débit
    w_switch_penalty: 0.1 # Pénalité changements
```

### Configuration Signalisation (`signals_lagos.yaml`)
```yaml
signals:
  timings:
    min_green: 15.0     # Vert minimum 15s (trafic dense)
    max_green: 90.0     # Vert maximum 90s
    yellow: 4.0         # Jaune 4s (sécurité piétons)
    all_red: 3.0        # Rouge général 3s
```

## 📊 Réseau Victoria Island

Le système modélise 2 intersections clés du corridor Victoria Island :

### Intersection 1 - Nœud 2339926113
- **Nord-Sud** : Akin Adesola Street (primary, 3 voies, 50 km/h)
- **Est-Ouest** : Adeola Odeku Street (secondary, 2 voies, 40 km/h)

### Intersection 2 - Nœud 95636900  
- **Nord-Sud** : Akin Adesola Street (primary, 3 voies, 50 km/h)
- **Est-Ouest** : Adeola Odeku Street (secondary, 2 voies, 40 km/h)

### 8 Branches de Trafic
```
intersection_1_north_in   -> Entrée Nord Intersection 1
intersection_1_south_in   -> Entrée Sud Intersection 1  
intersection_1_north_out  -> Sortie Nord Intersection 1
intersection_1_south_out  -> Sortie Sud Intersection 1
intersection_2_north_in   -> Entrée Nord Intersection 2
intersection_2_south_in   -> Entrée Sud Intersection 2
intersection_2_north_out  -> Sortie Nord Intersection 2
intersection_2_south_out  -> Sortie Sud Intersection 2
```

## 📈 Métriques et Évaluation

### Métriques Principales
- **Temps d'attente moyen** : Temps véhicules à l'arrêt
- **Longueur des files** : Nombre véhicules en attente
- **Débit** : Véhicules/heure traversant l'intersection
- **Nombre de changements** : Fréquence commutations phases

### Comparaison Performance
- **Agent DQN** vs **Baseline fixe** (cycles fixes)
- **Évaluation** : 10+ épisodes avec graines aléatoires
- **Stabilité** : Variance des performances

## 🧪 Tests et Validation

### Tests Unitaires
```bash
pytest tests/test_components.py -v
```

### Tests d'Intégration
```bash
python test_lagos.py
```

### Validation des Configurations
```bash
python validate.py
```

## 📋 Données d'Entrée

### Format des Données Corridor
Le fichier `fichier_de_travail_corridor.csv` contient :
- **Node_from/Node_to** : Identifiants nœuds intersection
- **Street_name** : Nom de rue
- **Highway** : Type de route (primary/secondary/tertiary)
- **Oneway** : Direction (yes/no)
- **Length_m** : Longueur segment en mètres

### Analyse Automatique
Le script `analyze_corridor.py` :
1. **Identifie** les intersections majeures
2. **Génère** la topologie réseau
3. **Crée** le fichier `network_real.yaml`
4. **Configure** les paramètres par type de route

## 🔄 Processus de Développement

### 1. Phase d'Analyse
- Analyse données corridor Victoria Island
- Identification intersections clés
- Caractérisation types de trafic

### 2. Phase d'Adaptation
- Création configurations spécifiques Lagos
- Calibrage paramètres trafic
- Ajustement fonction de récompense

### 3. Phase de Test
- Validation composants individuels
- Tests intégration complète
- Comparaison avec baselines

### 4. Phase d'Évaluation
- Entraînement modèles DQN
- Évaluation performances
- Analyse stabilité

## 🐛 Débogage et Diagnostic

### Logs et Diagnostics
```bash
# Logs détaillés pendant entraînement
python train.py --config lagos --use-mock --timesteps 1000 --verbose

# Test composants individuels
python test_lagos.py

# Validation configuration
python -c "from utils.config import load_config; print(load_config('configs/env_lagos.yaml'))"
```

### Problèmes Courants

1. **Erreur import modules**
   ```bash
   # Vérifier PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:./src"
   ```

2. **Configuration manquante**
   ```bash
   # Regénérer configs Lagos
   python adapt_lagos.py
   ```

3. **Erreur réseau réel**
   ```bash
   # Regénérer réseau Victoria Island
   python analyze_corridor.py
   ```

## 📊 Résultats Expérimentaux

### Performance Baseline
```
Agent DQN Lagos:
- Récompense moyenne: -0.01 ± 0.00
- Changements de phase: 90/épisode
- Convergence: ~1000 timesteps

Baseline Fixe:
- Changements de phase: 59/épisode
- Cycles fixes 60s/60s
```

### Observations
- **Agent adaptatif** : Plus de changements de phase (réactivité)
- **Timing Lagos** : Respect contraintes 15s-90s
- **Stabilité** : Faible variance sur 10 épisodes

## 🔮 Extensions Futures

### Intégrations Possibles
1. **SUMO** : Simulation trafic réaliste
2. **CARLA** : Environnement 3D avec véhicules autonomes
3. **Real-time data** : APIs trafic temps réel Lagos

### Améliorations Algorithmiques
1. **Multi-agent** : Coordination plusieurs intersections
2. **A3C/PPO** : Algorithmes plus avancés
3. **Transfer learning** : Adaptation autres villes

### Extensions Réseau
1. **Plus d'intersections** : Corridor complet Victoria Island
2. **Modes de transport** : Piétons, bus, BRT
3. **Optimisation réseau** : Coordination globale

## 📞 Support et Contribution

### Structure du Code
- **Modulaire** : Composants indépendants testables
- **Configurable** : Toutes les configurations externalisées
- **Extensible** : Interfaces claires pour extensions

### Tests de Régression
Avant toute modification majeure :
```bash
# Tests complets
python test_lagos.py
pytest tests/
python demo.py 1
python train.py --config lagos --use-mock --timesteps 100
```

### Documentation
- **Code documenté** : Docstrings Python standard
- **Configuration** : Commentaires YAML explicatifs  
- **Architecture** : Schémas et diagrammes

## 📄 Licence et Citation

Projet développé pour l'optimisation du trafic urbain à Lagos, Nigeria.

### Citation Suggérée
```bibtex
@software{lagos_traffic_rl_2025,
  title={Système de Contrôle de Signalisation par Apprentissage par Renforcement - Victoria Island Lagos},
  author={[Auteur]},
  year={2025},
  url={[URL_REPO]}
}
```

---

**Note** : Ce système est optimisé pour le contexte spécifique de Victoria Island Lagos mais peut être adapté à d'autres environnements urbains en modifiant les configurations appropriées.
