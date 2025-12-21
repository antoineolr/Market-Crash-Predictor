# Market Crash Predictor

Système de détection de régimes de marché basé sur les modèles de Markov cachés (HMM) pour l'optimisation du risque sur l'indice S&P 500.

## Présentation
Ce projet implémente une approche probabiliste pour identifier trois régimes de marché distincts. L'objectif est de générer des signaux de sortie de marché (passage en cash) lors des phases de forte instabilité pour protéger le capital et améliorer le rendement ajusté au risque.

## Méthodologie

### Architecture du modèle
Le système repose sur un **HMM Gaussien à 3 états** :
* **État 0 (Stable) :** Faible volatilité, rendements constants.
* **État 1 (Volatile) :** Incertitude et fluctuations accrues.
* **État 2 (Crise) :** Phases de forte correction ou krach boursier.

### Features Engineering
Six indicateurs techniques servent d'observations au modèle :
* **Volatilité glissante (20j) :** Écart-type mobile des rendements log.
* **Volatilité GARCH(1,1) :** Estimation de la volatilité conditionnelle.
* **Momentum (20j) :** Moyenne mobile des rendements.
* **RSI (14j) :** Relative Strength Index.
* **Volume Standardisé :** Z-score du volume de transactions (60j).
* **Ratio VIX/Vol :** Écart entre volatilité implicite et réalisée.

### Stratégie de Trading
* **Signaux :** Sortie du marché vers le cash lorsque $P(Crise)$ dépasse un seuil optimal.
* **Optimisation :** Seuil déterminé par maximisation du ratio de Sharpe sur la période 2000-2018.
* **Lissage :** Application d'une fenêtre modale de 5 jours pour réduire les faux signaux.

## Résultats (Période test 2019-2024)

| Métrique | Buy & Hold | Stratégie HMM | Amélioration |
| :--- | :--- | :--- | :--- |
| **Rendement total** | 108.3% | 114.6% | +5.8% |
| **Volatilité annuelle** | 20.2% | 12.8% | -36.5% |
| **Ratio de Sharpe** | 0.55 | 0.90 | +65.5% |
| **Max Drawdown** | -36.1% | -19.1% | -47.1% |

## Installation

### Prérequis
* Python 3.9 ou supérieur



Utilisation
Le script principal automatise l'ensemble du pipeline :

Téléchargement des données (S&P 500 & VIX).

Calcul des indicateurs techniques.

Entraînement et optimisation du seuil (2000-2018).

Backtesting et génération des signaux (2019-2024).

Visualisation des performances et des états cachés.

Bash

python main.py
Structure du projet
main.py : Script principal d'exécution.

config.py : Centralisation des paramètres (périodes, seuils, coûts).

models/ : Implémentation du HMM et de la logique GARCH.

utils/ : Fonctions de traitement de données et graphiques.

Dépendances
numpy, pandas, yfinance, arch, hmmlearn, scikit-learn, scipy, matplotlib, joblib.

### Installation des dépendances
```bash
pip install numpy pandas yfinance arch hmmlearn scikit-learn scipy matplotlib joblib
