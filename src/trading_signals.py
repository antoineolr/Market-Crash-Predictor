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

def backtest_threshold(data, probabilities, thresholds,tax,portfolio_init):
    market_returns=data['Log_Return']
    market_returns=market_returns.to_numpy()

    sharpe_ratios=[]
    cumuls=[]
    n_transactions=[]
    drawdowns=[]

    for threshold in thresholds:        
        signals=generate_signals(probabilities,threshold)

        n_transactions.append(count_transactions(signals))

        strategy_returns=np.zeros(len(signals))
        strategy_returns[0]=0

        strategy_returns_net = np.zeros(len(signals))
        strategy_returns_net[0]=0

        transition_cost=np.zeros(len(signals))
        transition_cost[0]=0
        
        portfolio_value=np.zeros(len(signals))
        portfolio_value[0]=portfolio_init

        for i in range(1,len(signals)):
            if signals[i]=="BUY" or signals[i]=="HOLD":
                strategy_returns[i]=market_returns[i]
            
            if signals[i] != signals[i-1]:
                transition_cost[i] = portfolio_value[i-1] * tax
            else:
                transition_cost[i] = 0
            
            portfolio_value[i]=portfolio_value[i-1] * (1+strategy_returns[i]) - transition_cost[i]
            strategy_returns_net[i]=(portfolio_value[i] - portfolio_value[i-1])/portfolio_value[i-1]
                  
        strategy_returns=np.array(strategy_returns)


        cumul=np.cumprod(strategy_returns_net+1) - 1
        
        running_max=np.maximum.accumulate(portfolio_value)
        drawdown=(portfolio_value - running_max)/running_max
        max_drawdown=np.min(drawdown)
        drawdowns.append(max_drawdown)
        
        if np.std(strategy_returns_net)==0:
            sharpe_ratio=np.nan
        else:
            sharpe_ratio=np.mean(strategy_returns_net)/np.std(strategy_returns_net) * np.sqrt(252)
        
        cumuls.append(cumul[-1])
        sharpe_ratios.append(sharpe_ratio)
    
    df=pd.DataFrame(index=thresholds)
    df['Sharpe_Ratio']=pd.Series(sharpe_ratios)
    df['Total_Returns']=pd.Series(cumuls)
    df['N_Transaction']=pd.Series(n_transactions)
    df['Max_Drawdown']=pd.Series(drawdowns)


    optimal_threshold=df['Sharpe_Ratio'].idxmax()
    return optimal_threshold








