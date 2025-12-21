Market Crash Predictor
Description
Ce projet implémente un système de détection de crises boursières basé sur les modèles de Markov cachés (HMM) pour générer des signaux de trading sur l'indice S&P 500. Le modèle identifie trois régimes de marché distincts (Stable, Volatile, Crise) et utilise ces états pour construire une stratégie de gestion de risque qui surpasse significativement une stratégie Buy & Hold classique.

Méthodologie
Architecture du modèle
Le système utilise un modèle de Markov caché gaussien à 3 états pour classifier les régimes de marché :

État 0 (Stable) : Conditions de marché normales avec faible volatilité
État 1 (Volatile) : Périodes d'incertitude et de fluctuations accrues
État 2 (Crise) : Phases de forte correction ou de krach boursier
Features engineering
Six indicateurs techniques sont calculés et utilisés comme observations pour le modèle HMM :

Rolling Volatility : Écart-type mobile des rendements logarithmiques (20 jours)
GARCH Volatility : Volatilité conditionnelle estimée par un modèle GARCH(1,1)
Momentum : Moyenne mobile des rendements (20 jours)
RSI : Relative Strength Index (14 jours)
Volume Standardized : Z-score du volume de transactions (60 jours)
VIX/Vol Ratio : Ratio entre l'indice VIX et la volatilité réalisée
Stratégie de trading
La stratégie génère des signaux binaires :

SELL : Sortie du marché vers le cash lorsque P(Crise) dépasse un seuil optimal
HOLD : Maintien de la position longue sur le marché
Le seuil optimal est déterminé par maximisation du ratio de Sharpe sur les données d'entraînement (période 2000-2018). Un lissage modal (fenêtre de 5 jours) est appliqué pour réduire les faux signaux.

Structure du projet
Installation
Prérequis
Python 3.9 ou supérieur
Installation des dépendances
Utilisation
Exécution du système complet
Le script effectue les opérations suivantes :

Téléchargement des données S&P 500 et VIX (2000-2024)
Calcul des features techniques
Entraînement du modèle HMM sur la période 2000-2018
Optimisation du seuil de décision
Génération des signaux de trading sur la période test 2019-2024
Backtesting et comparaison avec Buy & Hold
Visualisation des résultats (états cachés, probabilités, signaux, performances)
Configuration
Les paramètres sont centralisés dans config.py :

Périodes d'analyse et de split train/test
Fenêtres de calcul des indicateurs techniques
Paramètres du modèle HMM (nombre d'états, type de covariance, itérations)
Seuils de trading et coûts de transaction
Capital initial et taux sans risque
Résultats
Performance sur la période test (2019-2024)
Métrique	Buy & Hold	HMM Strategy	Amélioration
Rendement total	108.3%	114.6%	+5.8%
Rendement annuel	13.0%	13.6%	+4.3%
Volatilité annuelle	20.2%	12.8%	-36.5%
Ratio de Sharpe	0.55	0.90	+65.5%
Drawdown maximum	-36.1%	-19.1%	-47.1%
La stratégie HMM délivre des rendements supérieurs avec une volatilité et un drawdown significativement réduits, résultant en un ratio de Sharpe 65% plus élevé que le Buy & Hold.

Interprétation
Le modèle détecte efficacement les périodes de crise (notamment la crise COVID de mars 2020) et sort du marché au bon moment, permettant de :

Réduire l'exposition pendant les corrections majeures
Maintenir une volatilité faible du portefeuille
Améliorer le rendement ajusté au risque
Le système effectue environ 7 transactions sur la période test de 6 ans, évitant le sur-trading et minimisant les coûts de transaction.

Dépendances
numpy : Calculs numériques
pandas : Manipulation de données
yfinance : Téléchargement des données de marché
arch : Modèles GARCH pour la volatilité
hmmlearn : Modèles de Markov cachés
scikit-learn : Préprocessing (StandardScaler)
scipy : Fonctions statistiques
matplotlib : Visualisation
joblib : Sérialisation des modèles
