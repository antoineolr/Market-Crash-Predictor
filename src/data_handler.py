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


def compute_log_returns(data):
    """Calcule les log-rendements en Close-to-Close convention"""
    data['Log_Return']=np.log(data['Close']/data.shift(1)['Close'])
    data.dropna(inplace=True)
    return data


def align_datasets(sp500_data, vix_data):
    """Synchronise S&P 500 et VIX sur les mêmes dates"""
    sp500_data = sp500_data.rename(columns={
        'Open': 'SP500_Open',
        'High': 'SP500_High', 
        'Low': 'SP500_Low',
        'Close': 'SP500_Close',
        'Adj Close': 'SP500_Adj_Close',
        'Volume': 'SP500_Volume',
        'Log_Return': 'Log_Return' 
    })
    
    vix_data= vix_data.rename(columns={
        'Open': 'VIX_Open',
        'High': 'VIX_High',
        'Low': 'VIX_Low', 
        'Close': 'VIX_Close',
        'Adj Close': 'VIX_Adj_Close' ,
        'Volume': 'VIX_Volume'
    })
    result=sp500_data.join(vix_data,how='inner')
    return result


def preprocess_data(sp500_data, vix_data):
    """Pipeline complet de preprocessing"""
    sp500_data=compute_log_returns(sp500_data)
    result=align_datasets(sp500_data,vix_data)
    result.dropna(inplace=True)
    return result