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
    """returns: une série de log-retours"""
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
    delta=data['SP500_Close']-data['SP500_Close'].shift(1)
    profit=delta.where(delta>0,0)
    loss=(-delta).where(delta<0,0)
    mean_profit=profit.rolling(period).mean()
    mean_loss=loss.rolling(period).mean()
    RS=mean_profit/mean_loss
    RS = RS.replace([np.inf, -np.inf], np.nan) 
    RSI=100 - (100/(1+RS))
    data['RSI']=RSI
    return data

def compute_volume_standardized(data,window):
    mean_vol=data['SP500_Volume'].rolling(window).mean()
    std_vol=data['SP500_Volume'].rolling(window).std()
    data['Volume_Standardized']=(data['SP500_Volume'] - mean_vol)/std_vol
    data['Volume_Standardized']=data['Volume_Standardized'].replace([np.inf, -np.inf], np.nan)
    return data

def compute_vix_vol_ratio(data):
    """Le dataframe doit contenir VIX_Close et Rolling_Vol"""
    if 'VIX_Close' not in data.columns or 'Rolling_Vol' not in data.columns:
        print("⚠️ Colonnes VIX_Close ou Rolling_Vol manquantes")
        return data
    data['VIX_Vol_Ratio']=data['VIX_Close']/data['Rolling_Vol']
    data['VIX_Vol_Ratio']=data['VIX_Vol_Ratio'].replace([np.inf, -np.inf], np.nan)
    return data



def add_features(data,window_rolling_vol=20, window_momentum= 20,period=14,window_volume_std=60):
    data=compute_rolling_volatility(data,window_rolling_vol)
    fitted_model=fit_garch_model(data['Log_Return'])
    garch_model=extract_garch_volatility(fitted_model)
    data['GARCH_Vol']= garch_model
    data=compute_momentum(data,window_momentum)
    data=compute_rsi(data,period)
    data=compute_volume_standardized(data,window_volume_std)
    data=compute_vix_vol_ratio(data)

    data.dropna(inplace=True)
    return data



