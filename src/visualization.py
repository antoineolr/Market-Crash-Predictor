import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def graph_hidden_states(data, states):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index,data['SP500_Close'])
    plt.fill_between(data.index, data['SP500_Close'].min(), data['SP500_Close'].max(), 
                     where=(states==0), color='green', alpha=0.3, label='État Stable')
    plt.fill_between(data.index, data['SP500_Close'].min(), data['SP500_Close'].max(), 
                     where=(states==1), color='orange', alpha=0.3, label='État Volatile')
    plt.fill_between(data.index, data['SP500_Close'].min(), data['SP500_Close'].max(), 
                     where=(states==2), color='red', alpha=0.3, label='État Crise')
    plt.legend()
    plt.title('États Cachés du Marché')
    plt.xlabel('Date')
    plt.ylabel('Prix S&P 500')
    plt.show()

def graph_crisis_prob(data, probabilities, threshold):
    plt.figure(figsize=(12, 6))
    p_crisis = probabilities[:, 2]  # Extract crisis probabilities (column 2)
    plt.plot(data.index, p_crisis, color='red', label='P(Crise)')
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Seuil = {threshold}')
    plt.fill_between(data.index, 0, p_crisis, 
                     where=(p_crisis < threshold), color='green', alpha=0.3)
    plt.fill_between(data.index, 0, p_crisis, 
                     where=(p_crisis >= threshold), color='red', alpha=0.3)
    plt.legend()
    plt.title('Probabilité de Crise')
    plt.xlabel('Date')
    plt.ylabel('Probabilité')
    plt.ylim(0, 1)
    plt.show()


def graph_trading_signals(data, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['SP500_Close'],'k-')

    # Détection des signaux BUY explicites
    buy_signals_explicit=data.index[signals == "BUY"]
    
    # Détection des entrées sur le marché (transitions SELL → HOLD)
    buy_signals_transitions = []
    for i in range(1, len(signals)):
        if signals[i] == "HOLD" and signals[i-1] == "SELL":
            buy_signals_transitions.append(data.index[i])
    
    # Combiner les deux sources de signaux BUY
    all_buy_signals = buy_signals_explicit.union(buy_signals_transitions) if len(buy_signals_explicit) > 0 else buy_signals_transitions
    
    sell_signals=data.index[signals == "SELL"]

    if len(all_buy_signals) > 0:
        plt.scatter(all_buy_signals, data.loc[all_buy_signals, 'SP500_Close'], color='green', marker='^', label='BUY (Entrée)', s=200)
    plt.scatter(sell_signals, data.loc[sell_signals, 'SP500_Close'], color='red', marker='v', label='SELL (Sortie)', s=100)
    plt.legend()
    plt.title('Signaux de Trading')
    plt.xlabel('Date')
    plt.ylabel('Prix')

    plt.show()

    
def graph_comparison(data, portfolio_bh, portfolio_hmm):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, portfolio_bh, 'b-', label='Buy & Hold')
    plt.plot(data.index, portfolio_hmm, 'g-', label='HMM Strategy')
    plt.legend()
    plt.title('Comparaison des Stratégies')
    plt.xlabel('Date')
    plt.ylabel('Valeur du Portfolio')
    plt.show()


def graph_drawdown(data, dd_bh, dd_hmm):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, dd_bh, 'r-', label='Buy & Hold')
    plt.plot(data.index, dd_hmm, 'g-', label='HMM Strategy')
    plt.axhline(y=0,color='green')
    plt.legend()
    plt.title('Drawdown Comparaison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.show()

def hist_metrics(results_bh, results_hmm):
    metrics = ['total_return','annual_return','annual_volatility','sharpe','max_drawdown']
    bh_values = [results_bh[m] for m in metrics]
    hmm_values = [results_hmm[m] for m in metrics]
    
    x = range(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], bh_values, width, label='Buy & Hold')
    plt.bar([i + width/2 for i in x], hmm_values, width, label='HMM Strategy')
    plt.xlabel('Métriques')
    plt.ylabel('Valeurs')
    plt.title('Comparaison des Métriques de Performance')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()