import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

def load_market_data(ticker, start_date, end_date):
    """Télécharge les données via yfinance"""
    print(f"📊 Récupération des données de {ticker} prix de l'indice boursier entre {start_date} et {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        raise ValueError(f"Aucune donnée trouvée pour le ticker : {ticker}")
    
    return data


    
def load_vix_data(start_date, end_date):
    """Télécharge le VIX"""
    print(f"📊 Récupération des données des prix VIX de l'indice boursier entre {start_date} et {end_date}...")
    data = yf.download(tickers="^VIX",start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        raise ValueError(f"Aucune donnée trouvée")
    
    return data




def save_data_to_csv(data, filename):
    """Sauvegarde localement pour éviter de re-télécharger"""
    data.to_csv(filename, index=True)
    print(f"données sauvegardées correctement dans {filename}")