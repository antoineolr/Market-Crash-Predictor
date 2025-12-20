import numpy as np
import pandas as pd
from scipy.stats import mode


def generate_signals(probabilities,threshold):
    signals=np.where(probabilities[:,2]>threshold,"SELL","HOLD")
    signals=signals.tolist()
    for i in range(1,len(signals)):
        if signals[i]=="HOLD" and signals[i-1]=="SELL":
            signals[i]="BUY"
    signals=np.array(signals)
    return signals

def generate_signals_hysteresis(probabilities, upper_threshold, lower_threshold):
    p_crisis=np.array(probabilities[:,2])
    signals=np.zeros_like(p_crisis,dtype=str)
    signals[0]="HOLD"
    for i in range(1,len(signals)):
        if p_crisis[i]>upper_threshold:
            signals[i]="SELL"
        elif p_crisis[i]<lower_threshold and signals[i-1]=="SELL":
            signals[i]="BUY"
        elif p_crisis[i]<lower_threshold and signals[i-1]!="SELL":
            signals[i]="HOLD"
        else:
            signals[i]=signals[i-1]
    return signals

def apply_smoothing(signals,window):
    signals_df=pd.Series(signals)
    signals_df=signals_df.rolling(window, center=True).apply(lambda x: mode(x)[0][0])
    signals_df = signals_df.fillna(method='ffill')
    signals=signals_df.to_numpy()
    return signals

def add_transaction_costs(signals,transaction_amounts,tax):
    transaction_costs=np.zeros_like(transaction_amounts)
    signals=signals.tolist()
    for i in range(1,len(signals)):
        if signals[i]=="SELL" and signals[i-1]!="SELL":
            transaction_costs[i]=np.abs(transaction_amounts[i])*tax
        elif signals[i]!="SELL" and signals[i-1]=="SELL":
            transaction_costs[i]=np.abs(transaction_amounts[i])*tax
    return transaction_costs

def count_transactions(signals):
    signals=signals.tolist()
    count=0
    for i in range(1,len(signals)):
        if signals[i]=="SELL" and signals[i-1]!="SELL":
            count+=1
        ###depend de si on compte une transaction par cycle ou a chaque changement
        ###elif signals[i]!="SELL" and signals[i-1]=="SELL":
            ###count+=1
    return count

def backtest_threshold(probabilities, data, thresholds):




