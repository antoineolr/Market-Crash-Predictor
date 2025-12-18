import numpy as np
import pandas as pd
from arch import arch_model 

def compute_rolling_volatility(data,window):
    """
    Calcule la volatilité réalisée sur une fenêtre glissante.
    
    Args:
        data: DataFrame contenant la colonne 'Log_Return'
        window: Taille de la fenêtre (défaut: 20 jours)
    
    Returns:
        DataFrame avec la nouvelle colonne 'Rolling_Vol'
    """

    data['Rolling_Vol']=data['Log_Return'].rolling(window).std()
    return data

def fit_garch_model(returns):
    model = arch_model(returns*100, vol='Garch', p=1, q=1, mean='Zero')
    try:
        fitted_model = model.fit(disp='off', show_warning=False)
        return fitted_model
    except Exception as e:
        print(f"⚠️ Erreur GARCH: {e}")
        return None

def extract_garch_volatility(fitted_model):
    """fitted_model est le model fitté avec le GARCH"""
    if fitted_model is None:
        return None 
    
    garch_vol=(fitted_model.conditional_volatility)/100
    return garch_vol


def compute_momentum(data,window):
    """Le data doit contenir la colonne Log_Return"""
    data['Momentum']=data['Log_Return'].rolling(window).mean()
    return data

def compute_rsi(data, period):
    

def compute_volume_standardized():

def compute_vix_vol_ratio():

def add_features():


