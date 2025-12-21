import numpy as np
import pandas as pd

def calculate_portfolio_value(returns, initial_capital):
    return initial_capital * np.cumprod(1 + returns)

def calculate_metrics(returns, portfolio_values,risk_free_rate):
    if isinstance(portfolio_values, pd.Series):
        portfolio_array = portfolio_values.values
    else:
        portfolio_array = portfolio_values
    
    running_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    total_return = (portfolio_array[-1] / portfolio_array[0]) - 1
    
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_volatility = np.std(returns) * np.sqrt(252)
    
    if np.std(returns) == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def backtest_buy_and_hold(data, initial_capital):
    returns=data['Log_Return']
    portfolio_value=calculate_portfolio_value(returns,initial_capital)
    metrics=calculate_metrics(returns,portfolio_value,0.02)
    return metrics, portfolio_value

def backtest_hmm_strategy(data, signals, tax, initial_capital):
    """signal est composé de HOLD, BUY et SELL"""
    market_returns=data['Log_Return']
    market_returns=market_returns.to_numpy()

    strategy_returns=np.zeros(len(signals))
    strategy_returns[0]=0

    strategy_returns_net = np.zeros(len(signals))
    strategy_returns_net[0]=0

    transition_cost=np.zeros(len(signals))
    transition_cost[0]=0
    
    portfolio_value=np.zeros(len(signals))
    portfolio_value[0]=initial_capital

    for i in range(1,len(signals)):
        if signals[i]=="BUY" or signals[i]=="HOLD":
            strategy_returns[i]=market_returns[i]
        
        if signals[i] != signals[i-1]:
            transition_cost[i] = portfolio_value[i-1] * tax
        else:
            transition_cost[i] = 0
        
        portfolio_value[i]=portfolio_value[i-1] * (1+strategy_returns[i]) - transition_cost[i]
        strategy_returns_net[i]=(portfolio_value[i] - portfolio_value[i-1])/portfolio_value[i-1]

    metrics=calculate_metrics(strategy_returns_net,portfolio_value,0.02)
    return metrics, portfolio_value

def compare_strategies(results_bh, results_hmm):
    df=pd.DataFrame(index=['total_return','annual_return','annual_volatility','sharpe','max_drawdown'])
    df['Buy & Hold']=pd.Series(results_bh)
    df['HMM']=pd.Series(results_hmm)
    df['Difference']=df['HMM'] - df['Buy & Hold']
    d1=(results_hmm['total_return']-results_bh['total_return'])/results_bh['total_return'] * 100
    d2=(results_hmm['annual_return']-results_bh['annual_return'])/results_bh['annual_return'] * 100
    d3= - (results_hmm['annual_volatility']-results_bh['annual_volatility'])/results_bh['annual_volatility'] * 100
    d4= (results_hmm['sharpe']-results_bh['sharpe'])/results_bh['sharpe'] * 100
    d5= - (results_hmm['max_drawdown']-results_bh['max_drawdown'])/results_bh['max_drawdown'] * 100
    compar=[d1,d2,d3,d4,d5]
    df['Comparison']=pd.Series(compar, index=df.index)
    return df